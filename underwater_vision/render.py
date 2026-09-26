# -*- coding: utf-8 -*-
"""最常用入口：``WaterRenderer`` —— 一个"渲染出来就是水下图"的 Renderer 包装。

设计目标就是**易调用**：换掉 ``mujoco.Renderer(...)`` 这一行，其余代码不动。

    import mujoco
    from underwater_vision import WaterRenderer

    model = mujoco.MjModel.from_xml_path("scene.xml")
    data = mujoco.MjData(model)

    wr = WaterRenderer(model, 224, 224, params="turbid", seed=0)   # 一行
    img = wr.render(data, camera="grasp_cam")                      # 已经是水下图
    clean, wet = wr.render_pair(data, "grasp_cam")                 # 要对照就成对拿

已有 ``mujoco.Renderer`` 不想重建（会多开一个 GL 上下文）时用 ``wrap``：

    wr = WaterRenderer.wrap(my_renderer, params="coastal", seed=0)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from . import image as _image
from .params import WaterParams
from .scene import SceneStyle, apply_style, randomize_lights, snapshot_lights

__all__ = ["WaterRenderer", "save_png"]


def _resolve_params(params: Any) -> WaterParams:
    if params is None:
        return WaterParams.preset("clear")
    if isinstance(params, WaterParams):
        return params
    if isinstance(params, str):
        return WaterParams.preset(params)
    if isinstance(params, dict):
        return WaterParams.from_dict(params)
    raise TypeError(f"params 只能是 WaterParams / 预设名 / dict / None，收到 {type(params)}")


class WaterRenderer:
    """把 ``mujoco.Renderer`` 包成"输出即水下图"的渲染器。

    Args:
        model: ``mujoco.MjModel``。
        width, height: 输出尺寸（**注意与 ``mujoco.Renderer`` 的 (height, width) 相反**，
            这里按图像的习惯给 width/height，内部已转好）。
        params: :class:`WaterParams` / 预设名 / dict / None(=clear)。
        visibility: 便捷参数：用"能见度（米）"直接生成参数（会覆盖 ``params``）。
        seed: 随机种子（颗粒/噪声）。同一个 seed 得到同一组随机数序列。
        use_depth: 是否逐像素用深度算衰减（需要多渲一遍深度图，约慢一倍）。
            关掉则用 ``params.depth_default`` 的统一距离。
        fovy_deg: 覆盖相机视场角（默认按相机自动取 ``model.cam_fovy``）。
        style: 光照风格（``"surface"/"mid"/"deep"/"dark"`` 或 :class:`SceneStyle`）；
            给了就立刻写进模型（探照灯/环境光）。
        light_jitter_deg: 同时随机抖动场景里已有 ``<light>`` 的方向。
        max_geom: 传给 ``mujoco.Renderer``。
    """

    def __init__(self, model: Any, width: int = 224, height: int = 224, *,
                 params: Any = None, visibility: float | None = None, seed: int | None = None,
                 use_depth: bool = True, fovy_deg: float | None = None,
                 style: SceneStyle | str | None = None, light_jitter_deg: float = 0.0,
                 max_geom: int = 10000) -> None:
        import mujoco   # 局部 import：不用渲染功能的人可以只 import 参数/成像部分

        self._mj = mujoco
        self.model = model
        self.width, self.height = int(width), int(height)
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width,
                                        max_geom=max_geom)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_depth = bool(use_depth)
        self._fovy_override = fovy_deg
        self.params = (WaterParams.from_visibility(float(visibility)) if visibility
                       else _resolve_params(params))
        self.style: SceneStyle | None = None
        self.style_info: dict[str, Any] = {}
        self.native_fog = bool(hasattr(self.renderer.scene, "fogStart"))
        self._light_base = snapshot_lights(model)   # 灯光绝对赋值的基准（防累积相乘）
        self._closed = False

        if style is not None or light_jitter_deg > 0:
            self.set_style(style, light_jitter_deg=light_jitter_deg)

    # ------------------------------------------------------------------
    @classmethod
    def wrap(cls, renderer: Any, *, params: Any = None, visibility: float | None = None,
             seed: int | None = None, use_depth: bool = True,
             fovy_deg: float | None = None) -> "WaterRenderer":
        """复用**已有**的 ``mujoco.Renderer``（不新建 GL 上下文）。"""
        import mujoco

        self = cls.__new__(cls)
        self._mj = mujoco
        self.renderer = renderer
        self.model = renderer.model
        self.width, self.height = int(renderer.width), int(renderer.height)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.use_depth = bool(use_depth)
        self._fovy_override = fovy_deg
        self.params = (WaterParams.from_visibility(float(visibility)) if visibility
                       else _resolve_params(params))
        self.style = None
        self.style_info = {}
        self.native_fog = bool(hasattr(renderer.scene, "fogStart"))
        self._light_base = snapshot_lights(self.model)
        self._closed = False
        return self

    # ------------------------------------------------------------------
    # 参数 / 光照
    # ------------------------------------------------------------------
    def set_params(self, params: Any, *, resample_from: dict[str, Any] | None = None) -> WaterParams:
        """换一组水参数。``resample_from`` 直接透传给 :meth:`WaterParams.sample`。"""
        if resample_from is not None:
            self.params = WaterParams.sample(self.rng, **resample_from)
        else:
            self.params = _resolve_params(params)
        return self.params

    def resample(self, level: str = "medium", rng: Any = None) -> WaterParams:
        """域随机化：重新采一组水参数（每次 reset/每段 episode 调一次即可）。"""
        g = self.rng if rng is None else rng
        self.params = WaterParams.sample(g, level=level)
        return self.params

    def set_style(self, style: SceneStyle | str | None = None,
                  light_jitter_deg: float = 0.0, *, from_water: bool = False) -> dict[str, Any]:
        """把水下光照风格写进模型（渲染前生效，不需要重编译）。

        内部用构造时存的灯光快照做**绝对赋值**，所以可以反复调用不会累积。
        """
        if from_water:
            style = SceneStyle.from_water(self.params)
        if style is None:
            style = SceneStyle.preset("surface")
        self.style = style if isinstance(style, SceneStyle) else SceneStyle.preset(style)
        self.style_info = apply_style(self.model, self.style, rng=self.rng,
                                      light_jitter_deg=light_jitter_deg, base=self._light_base)
        return self.style_info

    def randomize_scene_lights(self, *, dir_jitter_deg: float = 10.0,
                               intensity_range: tuple[float, float] = (0.6, 1.2),
                               tint_jitter: float = 0.08, randomize_pos: float = 0.0
                               ) -> dict[str, Any]:
        """域随机化：抖动场景里已有 ``<light>`` 的方向/强度/位置（基于灯光快照，可反复调）。"""
        return randomize_lights(
            self.model, self.rng, dir_jitter_deg=dir_jitter_deg,
            intensity_range=intensity_range, tint_jitter=tint_jitter,
            randomize_pos=randomize_pos, base=self._light_base,
            scale=self.style.light_scale if self.style else 1.0,
            tint=self.style.light_tint if self.style else (1.0, 1.0, 1.0))

    # ------------------------------------------------------------------
    # 渲染
    # ------------------------------------------------------------------
    def clean(self, data: Any, camera: Any = -1) -> np.ndarray:
        """原始（清水）渲染，uint8 (H,W,3)。"""
        self.renderer.update_scene(data, camera=camera)
        return self.renderer.render().copy()

    def depth(self, data: Any, camera: Any = -1) -> np.ndarray:
        """米制深度图 (H,W) float32（MuJoCo 深度缓冲反投影，背景≈远裁剪面）。"""
        self.renderer.enable_depth_rendering()
        try:
            self.renderer.update_scene(data, camera=camera)
            dep = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        return dep

    def render(self, data: Any, camera: Any = -1, *, apply: bool = True,
               rng: Any = None, depth: np.ndarray | None = None) -> np.ndarray:
        """渲染一帧。``apply=False`` 时等价于原始 MuJoCo 渲染。"""
        clean = self.clean(data, camera)
        if not apply:
            return clean
        if depth is None and self.use_depth:
            depth = self.depth(data, camera)
        return self._wet(clean, depth, camera, rng)

    def render_pair(self, data: Any, camera: Any = -1, *,
                    rng: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """返回 ``(清水图, 水下图)``，深度只渲一次 —— 做对照消融最常用。"""
        clean = self.clean(data, camera)
        depth = self.depth(data, camera) if self.use_depth else None
        return clean, self._wet(clean, depth, camera, rng)

    def render_many(self, data: Any, cameras: Mapping[str, Any] | Sequence[str], *,
                    apply: bool = True, rng: Any = None) -> dict[str, np.ndarray]:
        """多路相机一次渲完。``cameras`` 可以是 ``{"top": "grasp_cam", ...}`` 或相机名列表。"""
        if isinstance(cameras, Mapping):
            items = list(cameras.items())
        else:
            items = [(str(c), c) for c in cameras]
        out: dict[str, np.ndarray] = {}
        for key, cam in items:
            out[key] = self.render(data, camera=cam, apply=apply, rng=rng)
        return out

    def render_many_pairs(self, data: Any, cameras: Mapping[str, Any] | Sequence[str], *,
                          rng: Any = None) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """多路相机的 ``(清水, 水下)`` 成对结果。"""
        if isinstance(cameras, Mapping):
            items = list(cameras.items())
        else:
            items = [(str(c), c) for c in cameras]
        return {key: self.render_pair(data, cam, rng=rng) for key, cam in items}

    # ------------------------------------------------------------------
    def _wet(self, clean: np.ndarray, depth: np.ndarray | None, camera: Any,
             rng: Any = None) -> np.ndarray:
        return _image.apply_water(
            clean, params=self.params, depth=depth,
            distance=None if depth is not None else self.params.depth_default,
            rng=self.rng if rng is None else rng,
            fovy_deg=self._fovy(camera),
        )

    def _fovy(self, camera: Any) -> float | None:
        if self._fovy_override is not None:
            return float(self._fovy_override)
        mj = self._mj
        cid = -1
        if isinstance(camera, str):
            cid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, camera)
        elif isinstance(camera, (int, np.integer)):
            cid = int(camera)
        elif isinstance(camera, mj.MjvCamera):
            if camera.type == mj.mjtCamera.mjCAMERA_FIXED:
                cid = int(camera.fixedcamid)
        if cid is None or cid < 0:
            # 自由相机用的是 vis.global_.fovy
            return float(self.model.vis.global_.fovy) if hasattr(self.model, "vis") else None
        return float(self.model.cam_fovy[cid])

    # ------------------------------------------------------------------
    def info(self) -> dict[str, Any]:
        """元信息：**写进 dataset meta / checkpoint 旁边**，保证训练与评测成像一致。"""
        return {
            "water_params": self.params.to_dict(),
            "water_hash": self.params.hash(),
            "water_summary": self.params.summary(),
            "seed": self.seed,
            "use_depth": self.use_depth,
            "native_fog_supported": self.native_fog,
            "size": [self.width, self.height],
            "style": self.style.to_dict() if self.style else None,
        }

    def close(self, close_renderer: bool = True) -> None:
        if self._closed:
            return
        if close_renderer:
            try:
                self.renderer.close()
            except Exception:
                pass
        self._closed = True

    def __enter__(self) -> "WaterRenderer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return (f"WaterRenderer({self.width}x{self.height}, depth={self.use_depth}, "
                f"{self.params.summary()})")


# ----------------------------------------------------------------------
def save_png(path: str | Path, img: np.ndarray) -> Path:
    """存 PNG（需要 PIL）。``img`` 是 (H,W,3) uint8 或 0~1 float。"""
    import numpy as np_  # noqa: F401  (保持显式依赖)
    from PIL import Image

    a = np.asarray(img)
    u8 = (np.clip(a, 0.0, 1.0) * 255).astype(np.uint8) if a.dtype != np.uint8 else a
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(u8).save(p)
    return p
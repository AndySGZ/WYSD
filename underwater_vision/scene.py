# -*- coding: utf-8 -*-
"""MuJoCo 场景侧的水下"风格"：探照灯 / 环境光 / 场景灯光 / 雾的 XML 片段。

**不 import mujoco**（只对 model 做鸭子类型访问），所以可以单独 import 来生成 XML 片段。

关于雾的重要事实（本机 mujoco 3.8.0 实测）
------------------------------------------
- ``model.vis.map.fogstart/fogend/haze``、``model.vis.rgba.fog/haze`` **仍然存在也能赋值**，
  但 classic renderer 已经不再读它们：``MjvScene`` 里没有 fog 字段，
  ``mjRND_FOG`` / ``mjRND_HAZE`` 打开也没有任何像素变化 → **改了没用**。
- 还生效的是**灯光**：``model.vis.headlight.{ambient,diffuse,specular,active}`` 与
  ``model.light_*``（pos/dir/diffuse/cutoff/attenuation/castshadow/active）。

所以本模组的策略是：**雾/偏色/背向散射一律在图像域做**（见 ``image.py``），
这里只负责"把灯打对"（水下的工作照度、探照灯热点、关灯巡检），
以及给需要写场景 XML 的人生成片段（``xml_visual_snippet`` / ``xml_floodlight_snippet``）。
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "SceneStyle",
    "snapshot_lights",
    "apply_style",
    "randomize_lights",
    "set_viewer_fog",
    "native_fog_supported",
    "xml_visual_snippet",
    "xml_floodlight_snippet",
    "describe_lights",
    "headlight_state",
]

Vec3 = Sequence[float]

# mjtLightType（不 import mujoco 也能用）
LIGHT_SPOT, LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_IMAGE = 0, 1, 2, 3


def _v3(x: Vec3) -> tuple[float, float, float]:
    return (float(x[0]), float(x[1]), float(x[2]))


# ----------------------------------------------------------------------
# 光照风格
# ----------------------------------------------------------------------
_HEADLIGHT_PRESETS: dict[str, dict[str, Any]] = {
    # 水面附近 / 清澈浅水：环境光占主导，探照灯只是补充
    "surface": dict(headlight_ambient=(0.30, 0.32, 0.30), headlight_diffuse=(0.45, 0.47, 0.45),
                    headlight_specular=(0.05, 0.05, 0.05), light_scale=1.0),
    # 中层（10~30m）：环境光明显衰减，开始依赖探照灯
    "mid": dict(headlight_ambient=(0.12, 0.14, 0.15), headlight_diffuse=(0.70, 0.72, 0.70),
                headlight_specular=(0.08, 0.08, 0.08), light_scale=0.6),
    # 深水：几乎没有环境光，全靠探照灯（热点 + 边缘暗区）
    "deep": dict(headlight_ambient=(0.04, 0.05, 0.06), headlight_diffuse=(0.95, 0.96, 0.92),
                 headlight_specular=(0.12, 0.12, 0.12), light_scale=0.35),
    # 关灯巡检：只剩极弱环境光
    "dark": dict(headlight_ambient=(0.02, 0.025, 0.03), headlight_diffuse=(0.02, 0.02, 0.02),
                 headlight_specular=(0.0, 0.0, 0.0), light_scale=0.0),
}


@dataclass
class SceneStyle:
    """水下的光照风格（只管灯，不管雾 —— 雾在图像域）。"""

    name: str = "surface"
    headlight_ambient: Vec3 = (0.30, 0.32, 0.30)
    headlight_diffuse: Vec3 = (0.45, 0.47, 0.45)
    headlight_specular: Vec3 = (0.05, 0.05, 0.05)
    headlight_active: bool = True
    light_scale: float = 1.0                 # 场景里已有 <light> 的强度整体缩放
    light_tint: Vec3 = (1.0, 1.0, 1.0)       # 灯光色偏（水中偏青绿）
    shadows: bool = True

    @classmethod
    def preset(cls, name: str, **overrides) -> "SceneStyle":
        if name not in _HEADLIGHT_PRESETS:
            raise KeyError(f"未知光照风格 {name!r}，可用：{sorted(_HEADLIGHT_PRESETS)}")
        s = cls(name=name, **_HEADLIGHT_PRESETS[name])
        return s.replace(**overrides) if overrides else s

    @classmethod
    def from_water(cls, params: Any, **overrides) -> "SceneStyle":
        """由水体参数（能见度）推光照风格：越深/越浊 → 环境光越弱、越依赖探照灯。"""
        vis = float(getattr(params, "visibility_m", 3.0))
        name = "surface" if vis >= 5.0 else ("mid" if vis >= 2.0 else "deep")
        s = cls.preset(name)
        # 水的颜色顺手作为灯光色偏（偏青绿）
        amb = np.asarray(getattr(params, "ambient", (0.05, 0.2, 0.24)), dtype=np.float32)
        tint = np.clip(0.75 + amb * 1.2, 0.3, 1.0)
        s = s.replace(light_tint=_v3(tint))
        return s.replace(**overrides) if overrides else s

    def replace(self, **kw) -> "SceneStyle":
        bad = set(kw) - {f.name for f in dataclasses.fields(self)}
        if bad:
            raise TypeError(f"未知字段：{sorted(bad)}")
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        for k in ("headlight_ambient", "headlight_diffuse", "headlight_specular", "light_tint"):
            d[k] = list(d[k])
        return d


def headlight_state(model: Any) -> dict[str, Any]:
    """读当前 headlight 状态（写日志/断言用）。"""
    hl = model.vis.headlight
    return {"active": int(hl.active),
            "ambient": list(np.asarray(hl.ambient, dtype=float).ravel()),
            "diffuse": list(np.asarray(hl.diffuse, dtype=float).ravel()),
            "specular": list(np.asarray(hl.specular, dtype=float).ravel())}


def snapshot_lights(model: Any) -> dict[str, Any]:
    """把模型里灯光的"原始强度"存一份。

    为什么需要：``apply_style`` / ``randomize_lights`` 是**按倍率缩放**灯光强度的，
    如果每次都从"当前值"再乘一遍，反复调用会不断累积（越调越暗/越亮）。
    所以要用这份快照当基准做绝对赋值。
    """
    n = int(getattr(model, "nlight", 0))
    return {
        "nlight": n,
        "diffuse": np.array(model.light_diffuse, dtype=np.float32).copy() if n else None,
        "ambient": np.array(model.light_ambient, dtype=np.float32).copy() if n else None,
        "castshadow": (np.array(model.light_castshadow, dtype=np.int32).copy()
                       if n and hasattr(model, "light_castshadow") else None),
    }


def apply_style(model: Any, style: SceneStyle | str | None = None, *, rng: Any = None,
                light_jitter_deg: float = 0.0, light_scale: float | None = None,
                base: dict[str, Any] | None = None) -> dict[str, Any]:
    """把光照风格写进 ``model``（**渲染前调用，不需要重新编译模型**）。

    注意：只改灯光，不改雾 —— 雾在 mujoco 3.8 的 classic renderer 里已经无效。

    Args:
        model: ``mujoco.MjModel``（鸭子类型，只要有 ``vis`` 与 ``light_*``）。
        style: :class:`SceneStyle` / 风格名（``"surface"``/``"mid"``/``"deep"``/``"dark"``）。
        rng: 给 ``light_jitter_deg > 0`` 时抖动方向用。
        light_jitter_deg: 已有 ``<light>`` 的方向随机抖动角度。
        light_scale: 覆盖 ``style.light_scale``。
        base: :func:`snapshot_lights` 的结果。**反复调用本函数时务必传**，
            否则灯光强度会在"当前值"上反复相乘而累积。

    Returns:
        实际写入的字典（可直接记进 dataset meta）。
    """
    if style is None:
        style = SceneStyle.preset("surface")
    elif isinstance(style, str):
        style = SceneStyle.preset(style)

    hl = model.vis.headlight
    hl.ambient[:] = _v3(style.headlight_ambient)
    hl.diffuse[:] = _v3(style.headlight_diffuse)
    hl.specular[:] = _v3(style.headlight_specular)
    if hasattr(hl, "active"):
        hl.active = 1 if style.headlight_active else 0

    n = int(getattr(model, "nlight", 0))
    scale = style.light_scale if light_scale is None else float(light_scale)
    tint = np.asarray(_v3(style.light_tint), dtype=np.float32)
    if n > 0:
        if base is None:
            base = snapshot_lights(model)
        d0 = base["diffuse"] if base.get("diffuse") is not None else np.asarray(model.light_diffuse)
        a0 = base["ambient"] if base.get("ambient") is not None else np.asarray(model.light_ambient)
        model.light_diffuse[:] = np.clip(np.asarray(d0, dtype=np.float32) * scale * tint[None, :],
                                         0.0, 1.0)
        model.light_ambient[:] = np.clip(np.asarray(a0, dtype=np.float32) * scale * tint[None, :],
                                         0.0, 1.0)
        if base.get("castshadow") is not None:
            model.light_castshadow[:] = base["castshadow"]
        if hasattr(model, "light_castshadow"):
            model.light_castshadow[:] = 1 if style.shadows else 0

    if light_jitter_deg > 0 and n > 0:
        randomize_lights(model, rng, dir_jitter_deg=light_jitter_deg,
                         intensity_range=(1.0, 1.0), tint_jitter=0.0, base=base,
                         scale=scale, tint=style.light_tint)

    return {"style": style.to_dict(), "light_scale_applied": scale,
            "headlight": headlight_state(model), "nlight": n}


def _rotate_vec(v: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues 旋转。"""
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    c, s = np.cos(angle), np.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * float(np.dot(axis, v)) * (1.0 - c)


def randomize_lights(model: Any, rng: Any = None, *, dir_jitter_deg: float = 10.0,
                     intensity_range: tuple[float, float] = (0.6, 1.2),
                     tint_jitter: float = 0.08, randomize_pos: float = 0.0,
                     base: dict[str, Any] | None = None, scale: float = 1.0,
                     tint: Vec3 | None = None) -> dict[str, Any]:
    """对模型里已有的 ``<light>`` 做域随机化（方向 / 强度 / 色偏 / 位置）。

    渲染前调用即可生效；``nlight == 0`` 时只返回空记录。
    ``base`` = :func:`snapshot_lights` 的结果，反复调用时务必传，避免强度累积相乘。
    """
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    n = int(getattr(model, "nlight", 0))
    if n <= 0:
        return {"nlight": 0, "note": "模型里没有 <light>，只改了 headlight"}

    if base is None:
        base = snapshot_lights(model)
    d0 = base["diffuse"] if base.get("diffuse") is not None else np.asarray(model.light_diffuse)
    d0 = np.asarray(d0, dtype=np.float32)
    t0 = np.asarray(_v3(tint) if tint is not None else (1.0, 1.0, 1.0), dtype=np.float32)

    jit = np.deg2rad(float(dir_jitter_deg))
    lo, hi = intensity_range
    dirs = []
    for i in range(n):
        d = np.asarray(model.light_dir[i], dtype=np.float32)
        nd = float(np.linalg.norm(d))
        if nd < 1e-6:
            d = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        else:
            d = d / nd
        axis = rng.normal(size=3)
        ang = float(rng.uniform(-jit, jit)) if jit > 0 else 0.0
        d = _rotate_vec(d, axis, ang).astype(np.float32)
        model.light_dir[i] = d / max(float(np.linalg.norm(d)), 1e-9)
        dirs.append(list(model.light_dir[i]))

        g = float(rng.uniform(lo, hi))
        val = d0[i] * g * float(scale) * t0
        if tint_jitter > 0:
            val = val * np.clip(1.0 + rng.normal(0.0, tint_jitter, size=3), 0.7, 1.3).astype(np.float32)
        model.light_diffuse[i] = np.clip(val, 0.0, 1.0)
        if randomize_pos > 0 and hasattr(model, "light_pos"):
            p = np.asarray(model.light_pos[i], dtype=np.float32)
            model.light_pos[i] = (p + rng.normal(0.0, randomize_pos, size=3)).astype(np.float32)

    return {"nlight": n, "dirs": dirs,
            "diffuse": [list(np.asarray(model.light_diffuse[i], dtype=float)) for i in range(n)]}


def describe_lights(model: Any) -> list[dict[str, Any]]:
    """列出模型里所有灯的参数（调试用）。"""
    n = int(getattr(model, "nlight", 0))
    out = []
    for i in range(n):
        out.append({
            "id": i,
            "pos": list(np.asarray(model.light_pos[i], dtype=float)),
            "dir": list(np.asarray(model.light_dir[i], dtype=float)),
            "type": int(model.light_type[i]),
            "active": int(model.light_active[i]),
            "diffuse": list(np.asarray(model.light_diffuse[i], dtype=float)),
            "cutoff": float(model.light_cutoff[i]),
            "exponent": float(model.light_exponent[i]),
            "attenuation": list(np.asarray(model.light_attenuation[i], dtype=float)),
            "castshadow": int(model.light_castshadow[i]),
        })
    return out


# ----------------------------------------------------------------------
# 雾（legacy，仅 viewer / 老版本 mujoco 有意义）
# ----------------------------------------------------------------------
def native_fog_supported(scene: Any) -> bool:
    """渲染器的 ``MjvScene`` 是否还有 fog 字段（mujoco 3.8 已经没有 → False）。"""
    return hasattr(scene, "fogStart") or hasattr(scene, "fogstart")


def set_viewer_fog(model: Any, params: Any = None, *, fogstart: float | None = None,
                   fogend: float | None = None, haze: float | None = None) -> dict[str, Any]:
    """设置 ``model.vis`` 里的雾参数。

    ⚠️ 在 mujoco ≥3.8 的 classic renderer 上**不产生任何画面变化**（已实测）。
    这里保留是为了：① 老版本 mujoco；② 交互式 viewer 的观感；③ 把"水的颜色"
    写进模型便于别人读。真正的雾请用 :func:`underwater_vision.apply_water`。
    """
    vis = model.vis
    if params is not None:
        amb = np.asarray(getattr(params, "ambient", (0.05, 0.20, 0.24)), dtype=np.float32)
        vis.rgba.fog[:] = (float(amb[0]), float(amb[1]), float(amb[2]), 1.0)
        vis.rgba.haze[:] = (float(amb[0]), float(amb[1]), float(amb[2]), 1.0)
        vis.map.haze = 0.3
        if fogend is None:
            fogend = float(np.clip(getattr(params, "depth_max", 8.0) / max(model.stat.extent, 1e-6),
                                   0.05, 50.0))
        if fogstart is None:
            fogstart = float(fogend * 0.25)
    if fogstart is not None:
        vis.map.fogstart = float(fogstart)
    if fogend is not None:
        vis.map.fogend = float(fogend)
    if haze is not None:
        vis.map.haze = float(haze)
    return {"fogstart": float(vis.map.fogstart), "fogend": float(vis.map.fogend),
            "haze": float(vis.map.haze), "rgba.fog": list(np.asarray(vis.rgba.fog, dtype=float)),
            "effective": False, "note": "mujoco>=3.8 classic renderer 忽略这些值"}


# ----------------------------------------------------------------------
# XML 片段（写给场景文件用）
# ----------------------------------------------------------------------
def xml_visual_snippet(style: SceneStyle | str | None = None, *, offwidth: int = 1280,
                       offheight: int = 800, znear: float = 0.02, zfar: float = 30.0,
                       shadowsize: int = 2048, fog: Sequence[float] | None = None,
                       include_fog: bool = False) -> str:
    """生成可直接粘进场景 XML 的 ``<visual>`` 片段。

    ``include_fog=True`` 才会写 fog/haze（mujoco ≥3.8 无效，默认不写，免得误导）。
    """
    if style is None:
        style = SceneStyle.preset("surface")
    elif isinstance(style, str):
        style = SceneStyle.preset(style)
    a = _v3(style.headlight_ambient)
    d = _v3(style.headlight_diffuse)
    s = _v3(style.headlight_specular)
    lines = [
        "  <visual>",
        f'    <global offwidth="{offwidth}" offheight="{offheight}"/>',
        f'    <headlight ambient="{a[0]:.3f} {a[1]:.3f} {a[2]:.3f}"'
        f' diffuse="{d[0]:.3f} {d[1]:.3f} {d[2]:.3f}"'
        f' specular="{s[0]:.3f} {s[1]:.3f} {s[2]:.3f}"/>',
        f'    <quality shadowsize="{shadowsize}"/>',
        f'    <map znear="{znear}" zfar="{zfar}"/>',
    ]
    if include_fog:
        f0 = _v3(fog) if fog is not None else (0.05, 0.20, 0.24)
        lines.insert(4, f'    <map fogstart="0.25" fogend="2.0" haze="0.3"/>')
        lines.insert(5, f'    <rgba fog="{f0[0]:.3f} {f0[1]:.3f} {f0[2]:.3f} 1"'
                        f' haze="{f0[0]:.3f} {f0[1]:.3f} {f0[2]:.3f} 1"/>')
        lines.append("    <!-- 注意：mujoco>=3.8 的 classic renderer 已忽略 fog/haze，"
                     "水体成像请用 underwater_vision.apply_water -->")
    lines.append("  </visual>")
    return "\n".join(lines)


def xml_floodlight_snippet(name: str = "flood", pos: Sequence[float] = (1.0, -0.6, 1.2),
                           dir: Sequence[float] = (-0.3, 0.5, -1.0), target: str | None = None,
                           cutoff: float = 35.0, exponent: float = 8.0,
                           attenuation: Sequence[float] = (1.0, 0.0, 0.3),
                           diffuse: Sequence[float] = (0.85, 0.88, 0.80),
                           castshadow: bool = True) -> str:
    """生成一盏"ROV 探照灯"的 ``<light>`` 片段（聚光 + 衰减 + 阴影）。"""
    p, d, at, df = _v3(pos), _v3(dir), _v3(attenuation), _v3(diffuse)
    tgt = f' mode="targetbody" target="{target}"' if target else ""
    return (f'  <light name="{name}" pos="{p[0]} {p[1]} {p[2]}" dir="{d[0]} {d[1]} {d[2]}"{tgt}\n'
            f'         cutoff="{cutoff}" exponent="{exponent}"'
            f' attenuation="{at[0]} {at[1]} {at[2]}"\n'
            f'         diffuse="{df[0]:.3f} {df[1]:.3f} {df[2]:.3f}"'
            f' specular="0.12 0.12 0.12" castshadow="{"true" if castshadow else "false"}"/>')
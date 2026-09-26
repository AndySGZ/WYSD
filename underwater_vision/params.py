# -*- coding: utf-8 -*-
"""水下视觉参数：水体光学 + 传感器噪声 + 预设 + 域随机化采样。

这个文件**不依赖 mujoco**（只依赖 numpy），可以单独 import 来算参数、存/读 json。

核心概念
--------
- ``beta``         : 分通道衰减系数 (R,G,B)，单位 1/m。红>绿>蓝 —— 水下"红色先消失"。
- ``ambient``      : 无限远水色 / 背向散射色（悬浮颗粒被照亮后的颜色）。
- ``depth_max``    : 超过这个距离一律按该值算（避免远裁剪面 72m 把画面糊死）。
- ``visibility``   : 用户最关心的"能见度"（米）。用 ``from_visibility()`` 由它反推 beta。

用法
----
    from underwater_vision import WaterParams

    p = WaterParams.preset("turbid")            # 预设
    p = WaterParams.from_visibility(1.5)        # 用"能见度 1.5m"描述
    p = WaterParams.sample(level="medium")      # 域随机化采样一组
    print(p.to_dict());  p.to_json("water.json")
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

__all__ = ["WaterParams", "PRESETS", "VISIBILITY_LEVELS", "preset_names"]

Vec3 = Sequence[float]


def _v3(x: Vec3) -> tuple[float, float, float]:
    """把任意长度 >=3 的序列规整成 3 元组。"""
    return (float(x[0]), float(x[1]), float(x[2]))


# ----------------------------------------------------------------------
# 预设：清 / 沿岸 / 浑浊 / 深海 / 港口
# beta 的单位是 1/m。参考量级：清水开阔海域 0.05~0.2，沿岸 0.3~0.8，港口/浑浊 1.5~3。
# ----------------------------------------------------------------------
PRESETS: dict[str, dict[str, Any]] = {
    # 清水：只有轻微偏色，用于"接近空气"的对照
    "clear": dict(
        beta=(0.20, 0.07, 0.05), ambient=(0.02, 0.07, 0.09), beta_scatter=None,
        depth_max=8.0, scatter_sigma=0.6, scatter_dist=4.0,
        noise_sigma=2.0, shot_noise=0.02, particle_density=0.10,
        particle_brightness=0.25, vignette=0.05, exposure=1.0,
    ),
    # 沿岸：典型蓝绿，能见度 3~5m
    "coastal": dict(
        beta=(0.50, 0.18, 0.12), ambient=(0.05, 0.20, 0.24), beta_scatter=None,
        depth_max=8.0, scatter_sigma=1.4, scatter_dist=3.5,
        noise_sigma=3.5, shot_noise=0.04, particle_density=0.35,
        particle_brightness=0.35, vignette=0.12, exposure=0.98,
    ),
    # 浑浊：能见度 ~1.5m，红通道几乎全丢
    "turbid": dict(
        beta=(1.40, 0.55, 0.40), ambient=(0.08, 0.25, 0.28), beta_scatter=None,
        depth_max=4.0, scatter_sigma=2.6, scatter_dist=2.0,
        noise_sigma=6.0, shot_noise=0.07, particle_density=1.10,
        particle_brightness=0.45, vignette=0.20, exposure=0.92,
    ),
    # 深海：很蓝、很暗，靠探照灯
    "deep": dict(
        beta=(0.45, 0.12, 0.06), ambient=(0.01, 0.04, 0.08), beta_scatter=None,
        depth_max=10.0, scatter_sigma=1.0, scatter_dist=5.0,
        noise_sigma=5.0, shot_noise=0.06, particle_density=0.20,
        particle_brightness=0.30, vignette=0.30, exposure=0.85,
    ),
    # 港口/泥沙：能见度 <1m，泥沙悬浮
    "harbor": dict(
        beta=(2.50, 1.20, 0.90), ambient=(0.10, 0.22, 0.22), beta_scatter=None,
        depth_max=3.0, scatter_sigma=3.5, scatter_dist=1.5,
        noise_sigma=8.0, shot_noise=0.09, particle_density=2.00,
        particle_brightness=0.55, vignette=0.25, exposure=0.90,
    ),
}

# 能见度档位（米）—— 与设计文档里"能见度 3 档"对齐
VISIBILITY_LEVELS: dict[str, float] = {
    "clear": 8.0,      # 清
    "medium": 3.0,     # 中
    "turbid": 1.5,     # 浊
    "murky": 0.8,      # 极浊
}

# 域随机化的默认范围（起点，按需要改）。level 只用来缩放整体难度。
SAMPLE_RANGES: dict[str, tuple[float, float]] = {
    "visibility": (1.2, 6.0),
    "noise_sigma": (1.5, 6.0),
    "shot_noise": (0.0, 0.07),
    "particle_density": (0.05, 1.0),
    "vignette": (0.0, 0.25),
    "exposure": (0.88, 1.05),
    "scatter_sigma": (0.5, 2.8),
}
_LEVEL_SCALE = {"easy": 0.5, "medium": 1.0, "hard": 1.4}


def preset_names() -> list[str]:
    """可用预设名。"""
    return sorted(PRESETS)


@dataclass
class WaterParams:
    """一组水下视觉参数。所有距离单位都是米，图像噪声按 0~1 归一化尺度给。"""

    name: str = "custom"

    # --- 水体光学 ---
    beta: Vec3 = (0.20, 0.07, 0.05)          # 分通道衰减 (R,G,B) 1/m
    beta_scatter: Vec3 | None = None          # 背向散射系数；None -> 跟随 beta
    ambient: Vec3 = (0.02, 0.07, 0.09)        # 无限远水色 / 背向散射色
    depth_default: float = 1.5                # 没有深度图时用的统一距离 (m)
    depth_min: float = 0.15                   # 深度下限（避免贴脸时 T->1 又被噪声吞掉）
    depth_max: float = 8.0                    # 深度上限（远背景截断到这里）
    depth_scale: float = 1.0                  # 深度图整体缩放（相机/尺度标定用）
    radial_depth: bool = True                 # 把"沿光轴的 z 深度"修正成"径向距离"

    # --- 前向散射 ---
    scatter_sigma: float = 0.0                # 最大高斯模糊 σ(px)，0 = 关
    scatter_dist: float = 3.0                 # 达到最大模糊的距离 (m)

    # --- 传感器 / 相机 ---
    exposure: float = 1.0                     # 曝光增益
    gamma: float = 1.0                        # 幂律校正
    wb_gain: Vec3 = (1.0, 1.0, 1.0)           # 白平衡漂移（自动白平衡在蓝绿环境失准）
    vignette: float = 0.0                     # 暗角强度 0~1
    noise_sigma: float = 0.0                  # 高斯读出噪声（/255 尺度）
    shot_noise: float = 0.0                   # 散粒噪声强度（∝ sqrt(亮度)）
    jpeg_quality: int = 0                     # 0 = 不做；需要 PIL

    # --- 悬浮颗粒（海雪 / 泥沙）---
    particle_density: float = 0.0             # 每千像素的颗粒数
    particle_brightness: float = 0.35

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @classmethod
    def preset(cls, name: str, **overrides) -> "WaterParams":
        """按预设名构造，可再覆盖任意字段：``WaterParams.preset("turbid", noise_sigma=10)``。"""
        if name not in PRESETS:
            raise KeyError(f"未知预设 {name!r}，可用：{preset_names()}")
        p = cls(name=name, **PRESETS[name])
        return p.replace(**overrides) if overrides else p

    @classmethod
    def from_visibility(cls, visibility_m: float, ambient: Vec3 = (0.05, 0.20, 0.24),
                        **overrides) -> "WaterParams":
        """用"能见度（米）"描述水体：距离达到 ``visibility_m`` 时对比度降到 ~5%。

        beta_R = 3/visibility，G/B 按 0.35 / 0.25 比例衰减得更慢（红光先没）。
        """
        if visibility_m <= 0:
            raise ValueError("visibility_m 必须 > 0")
        v = float(visibility_m)
        p = cls(
            name=f"visibility{v:g}m",
            beta=(3.0 / v, 3.0 / v * 0.35, 3.0 / v * 0.25),
            ambient=_v3(ambient),
            depth_max=float(np.clip(4.0 * v, 1.0, 30.0)),
            scatter_sigma=float(np.clip(2.4 / max(v, 0.5), 0.4, 4.0)),
            scatter_dist=max(1.0, v),
            noise_sigma=float(np.clip(12.0 / v, 1.0, 10.0)),
            shot_noise=0.05,
            particle_density=float(np.clip(1.2 / v, 0.05, 2.5)),
            particle_brightness=0.4,
            vignette=0.15,
        )
        return p.replace(**overrides) if overrides else p

    @classmethod
    def sample(cls, rng: np.random.Generator | int | None = None,
               level: str = "medium", **overrides) -> "WaterParams":
        """域随机化采样一组参数。

        level: ``easy`` / ``medium`` / ``hard``，只缩放难度（能见度越低、噪声越大）。
        """
        if not isinstance(rng, np.random.Generator):
            rng = np.random.default_rng(rng)
        s = _LEVEL_SCALE.get(level, 1.0)

        vis_hi, vis_lo = SAMPLE_RANGES["visibility"]
        vis = float(rng.uniform(vis_hi, vis_lo)) / s          # 难度越高，能见度越低
        p = cls.from_visibility(max(vis, 0.6))

        def _lo_hi(key: str, scale: float = 1.0) -> float:
            lo, hi = SAMPLE_RANGES[key]
            return float(rng.uniform(lo, hi) * scale)

        hue = rng.uniform(-0.04, 0.04, size=3)
        p.name = f"sample[{level}]"
        p.ambient = _v3(np.clip(np.array(p.ambient) + hue, 0.0, 1.0))
        p.noise_sigma = min(_lo_hi("noise_sigma", s), 12.0)
        p.shot_noise = _lo_hi("shot_noise", s)
        p.particle_density = _lo_hi("particle_density", s)
        p.vignette = _lo_hi("vignette", s)
        p.exposure = _lo_hi("exposure")
        p.scatter_sigma = _lo_hi("scatter_sigma", s)
        p.wb_gain = _v3(np.clip(1.0 + rng.uniform(-0.08, 0.08, size=3), 0.8, 1.2))
        return p.replace(**overrides) if overrides else p

    # ------------------------------------------------------------------
    # 派生量 / 工具
    # ------------------------------------------------------------------
    @property
    def scatter_beta(self) -> tuple[float, float, float]:
        """背向散射系数（默认与衰减系数相同）。"""
        return _v3(self.beta_scatter) if self.beta_scatter is not None else _v3(self.beta)

    @property
    def beta_rgb(self) -> np.ndarray:
        return np.asarray(_v3(self.beta), dtype=np.float32)

    @property
    def ambient_rgb(self) -> np.ndarray:
        return np.asarray(_v3(self.ambient), dtype=np.float32)

    @property
    def wb_rgb(self) -> np.ndarray:
        return np.asarray(_v3(self.wb_gain), dtype=np.float32)

    @property
    def visibility_m(self) -> float:
        """等效能见度估计：beta_R 决定的距离（对比度 e^-3 ≈ 5%）。"""
        br = max(_v3(self.beta)[0], 1e-6)
        return float(3.0 / br)

    def replace(self, **kw) -> "WaterParams":
        """返回改了若干字段的新实例（dataclass 是可变/不可变都行，这里走 copy）。"""
        bad = set(kw) - {f.name for f in dataclasses.fields(self)}
        if bad:
            raise TypeError(f"未知字段：{sorted(bad)}")
        return dataclasses.replace(self, **kw)

    def to_dict(self) -> dict[str, Any]:
        """可 json 序列化的 dict（写进 dataset meta / checkpoint 旁边）。"""
        d = asdict(self)
        for k in ("beta", "beta_scatter", "ambient", "wb_gain"):
            if d[k] is not None:
                d[k] = list(d[k])
        d["_visibility_m"] = round(self.visibility_m, 4)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WaterParams":
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        return cls(**d)

    def to_json(self, path: str | Path | None = None) -> str:
        s = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if path is not None:
            Path(path).write_text(s, encoding="utf-8")
        return s

    @classmethod
    def from_json(cls, path: str | Path) -> "WaterParams":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def hash(self) -> str:
        """参数指纹：采集端与评测端比对，防止"两处成像不一致"的实验事故。"""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    def summary(self) -> str:
        """一行人类可读摘要。"""
        return (f"[{self.name}] 能见度≈{self.visibility_m:.2f}m  "
                f"beta={tuple(round(b, 3) for b in _v3(self.beta))}  "
                f"noise={self.noise_sigma:.1f}  颗粒={self.particle_density:.2f}  "
                f"模糊σ={self.scatter_sigma:.1f}px")
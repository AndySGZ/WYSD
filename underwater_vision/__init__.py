# -*- coding: utf-8 -*-
"""``underwater_vision`` —— MuJoCo 水下视觉渲染模组（图像域）。

一句话：**把 MuJoCo 渲出来的"清水图"过一遍水下成像模型，得到"水下图"**，
并且提供水下的灯光（探照灯/环境光）设置与域随机化采样。

两行用法
--------
    from underwater_vision import WaterRenderer

    wr = WaterRenderer(model, 224, 224, params="turbid", seed=0)
    img = wr.render(data, camera="grasp_cam")        # 已经是水下图

要清水/水下对照：
    clean, wet = wr.render_pair(data, "grasp_cam")

只想处理已有图像（不需要 mujoco）：
    from underwater_vision import apply_water, WaterParams
    wet = apply_water(clean_rgb, WaterParams.preset("coastal"), depth=depth_map)

为什么在图像域做（本机 mujoco 3.8 实测结论）
-------------------------------------------
- classic renderer 的 fog/haze **已失效**：``MjvScene`` 没有 fog 字段，
  ``mjRND_FOG``/``mjRND_HAZE`` 打开也无像素变化 → 渲染器只能给清水图；
- 渲染器的雾也无法表达"红光先丢 / 背向散射 / 颗粒 / 传感器噪声"；
- VLA 吃的是 2D 图像，图像域的域随机化自由度最大、最便宜、训练与评测最一致。

模块划分
--------
- :mod:`underwater_vision.params`  —— 参数、预设、能见度换算、域随机化采样（只依赖 numpy）
- :mod:`underwater_vision.image`   —— 成像模型 ``apply_water``（只依赖 numpy）
- :mod:`underwater_vision.scene`   —— MuJoCo 灯光风格 + XML 片段（不 import mujoco）
- :mod:`underwater_vision.render`  —— ``WaterRenderer``（包 ``mujoco.Renderer``）
- :mod:`underwater_vision.demo`    —— 自检 demo：``python -m underwater_vision.demo``
"""
from __future__ import annotations

from typing import Any

from .image import apply_water, distance_map, gaussian_blur, make_water_fn, radial_factor
from .params import PRESETS, VISIBILITY_LEVELS, WaterParams, preset_names
from .scene import (
    SceneStyle,
    apply_style,
    describe_lights,
    headlight_state,
    native_fog_supported,
    randomize_lights,
    set_viewer_fog,
    xml_floodlight_snippet,
    xml_visual_snippet,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # 参数
    "WaterParams", "PRESETS", "VISIBILITY_LEVELS", "preset_names",
    # 成像
    "apply_water", "make_water_fn", "distance_map", "radial_factor", "gaussian_blur",
    # 场景灯光 / XML
    "SceneStyle", "apply_style", "randomize_lights", "set_viewer_fog",
    "native_fog_supported", "describe_lights", "headlight_state",
    "xml_visual_snippet", "xml_floodlight_snippet",
    # 渲染（惰性导入，避免没装 mujoco 时 import 失败）
    "WaterRenderer", "save_png",
]

# 惰性属性：`import underwater_vision` 不需要 mujoco；用到 WaterRenderer 才导入
_LAZY_ATTRS = {"WaterRenderer": "render", "save_png": "render"}


def __getattr__(name: str) -> Any:
    if name in _LAZY_ATTRS:
        import importlib

        mod = importlib.import_module(f".{_LAZY_ATTRS[name]}", __name__)
        val = getattr(mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
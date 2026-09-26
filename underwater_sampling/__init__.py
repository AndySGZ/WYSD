# -*- coding: utf-8 -*-
"""采样任务族：分级场景 + 自动判据（**不提供专家策略**，数据由人手操作采集）。

    from underwater_sampling import SamplingEnv, LEVEL_ORDER, describe_all

    env = SamplingEnv("l2_gray_rock_random", seed=0)
    env.reset()
    print(env.instruction, env.metrics())
"""
from __future__ import annotations

from .tasks import LEVELS, LEVEL_ORDER, SURFACE_Z, describe_all, get

__version__ = "0.1.0"
__all__ = ["LEVELS", "LEVEL_ORDER", "SURFACE_Z", "SamplingEnv", "get", "describe_all"]


def __getattr__(name):
    """`SamplingEnv` 惰性导入：这样 `import underwater_sampling` 本身不需要 mujoco。"""
    if name == "SamplingEnv":
        from .env import SamplingEnv

        return SamplingEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

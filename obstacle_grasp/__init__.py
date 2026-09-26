# -*- coding: utf-8 -*-
"""障碍抓取族：分级场景 + 判据（**不做水下渲染**，数据由人手操作采集）。

与 `underwater_sampling` 同级、正交：那一族考"看不清"，这一族考"看不见 / 够不着 / 路被堵"。

    from obstacle_grasp import ObstacleGraspEnv, LEVEL_ORDER, describe_all

    env = ObstacleGraspEnv("o3_front_overhang", seed=0)
    env.reset()
    print(env.instruction, env.metrics())

展示图片：`python -m obstacle_grasp.shots`
"""
from __future__ import annotations

from .tasks import LEVELS, LEVEL_ORDER, TABLE_Z, describe_all, get

__version__ = "0.1.0"
__all__ = ["LEVELS", "LEVEL_ORDER", "TABLE_Z", "ObstacleGraspEnv", "get", "describe_all"]


def __getattr__(name):
    """`ObstacleGraspEnv` 惰性导入：`import obstacle_grasp` 本身不需要 mujoco。"""
    if name == "ObstacleGraspEnv":
        from .env import ObstacleGraspEnv

        return ObstacleGraspEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

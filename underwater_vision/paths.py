# -*- coding: utf-8 -*-
"""路径探测：定位 UranUS demo 仓库与场景 XML。

`underwater_vision` 会被放在两个位置，不能写死相对层级：

1. 仓库外：  ``E:/projects/mujoco/underwater_vision``        （``PYTHONPATH=E:/projects/mujoco``）
2. 仓库根内：``E:/projects/mujoco/WYSD/underwater_vision``    （作为 ``AndySGZ/WYSD`` 的一部分）

所以这里从本文件所在目录**向上逐层探测**，兼容两种放法；也可以用环境变量
``UNDERWATER_VISION_DEMO_DIR`` 直接指定 demo 仓库位置。
"""
from __future__ import annotations

import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent

DEMO_DIRNAME = "Lerobot-Uranus-VLA-Demo"
GRASP_SCENE = ("asset", "scene_uranus_grasp.xml")
TUTORIAL_SCENE = ("Lerobot-MujoCo-VLA-Tutorial", "asset", "scene_table.xml")
ENV_VAR = "UNDERWATER_VISION_DEMO_DIR"

__all__ = ["DEMO_DIR", "find_demo_dir", "scene_candidates", "ENV_VAR"]


def _bases(max_up: int = 4):
    """本模块目录、父目录、祖父目录……（上限 max_up 层）。"""
    b = _HERE
    for _ in range(max_up + 1):
        yield b
        if b.parent == b:
            break
        b = b.parent


def find_demo_dir(max_up: int = 4) -> Path | None:
    """找到 ``Lerobot-Uranus-VLA-Demo`` 的绝对路径；找不到返回 None。"""
    env = os.environ.get(ENV_VAR)
    if env and Path(env).is_dir():
        return Path(env).resolve()
    for base in _bases(max_up):
        for rel in (("WYSD", DEMO_DIRNAME), (DEMO_DIRNAME,)):
            cand = base.joinpath(*rel)
            if cand.joinpath(*GRASP_SCENE).is_file():
                return cand
    return None


def scene_candidates(max_up: int = 4) -> tuple[Path, ...]:
    """候选场景 XML（按优先级去重）。第一个存在的会被 demo 用上。"""
    out: list[Path] = []
    for base in _bases(max_up):
        out.append(base.joinpath("WYSD", DEMO_DIRNAME, *GRASP_SCENE))
        out.append(base.joinpath(DEMO_DIRNAME, *GRASP_SCENE))
        out.append(base.joinpath(*TUTORIAL_SCENE))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return tuple(uniq)


DEMO_DIR = find_demo_dir()
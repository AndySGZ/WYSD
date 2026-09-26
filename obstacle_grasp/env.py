# -*- coding: utf-8 -*-
"""障碍抓取族的运行环境：`ObstacleGraspEnv`。

只做三件事，不提供任何专家策略（数据由人手操作采集）：
  1. 按等级加载场景；
  2. 布局随机化（障碍物随目标一起移动，保证难度关系恒定）；
  3. 判据（沿用底座 UranusGraspEnv.success + 各等级的附加判据）。

复用 `UranusGraspEnv` 底座：6 自由度 IK、位置伺服残差闭环、夹爪宽度标定、慢速释放。
命名沿用底座：被操作物叫 `object`（带 `object_joint`/`object_geom`），目标点载体叫
`place_target`（带 `place_target_site`）。**本族不做水下渲染**，所以不依赖 underwater_vision。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from .tasks import OBJ_HALF, TABLE_Z, get

_HERE = Path(__file__).resolve().parent


def _find_demo_dir() -> Path | None:
    """定位 Lerobot-Uranus-VLA-Demo（复用 underwater_vision 的探测逻辑，找不到就自己向上找）。"""
    try:
        from underwater_vision.paths import find_demo_dir

        return find_demo_dir()
    except Exception:
        pass
    for base in (_HERE, *_HERE.parents):
        for rel in (("WYSD", "Lerobot-Uranus-VLA-Demo"), ("Lerobot-Uranus-VLA-Demo",)):
            cand = base.joinpath(*rel)
            if (cand / "src" / "env_grasp.py").is_file():
                return cand
    return None


_DEMO_DIR = _find_demo_dir()
if _DEMO_DIR is not None and str(_DEMO_DIR) not in sys.path:
    sys.path.insert(0, str(_DEMO_DIR))

try:
    from src.env_grasp import UranusGraspEnv
except Exception as exc:  # pragma: no cover
    raise ImportError(f"找不到 Lerobot-Uranus-VLA-Demo（Uranus 底座）：{exc}") from exc


class ObstacleGraspEnv(UranusGraspEnv):
    """一个等级的障碍抓取场景。

        env = ObstacleGraspEnv("o3_front_overhang", seed=0)
        env.reset()                     # 内部自己抽一组布局
        ... 人手操作 ...
        env.success(); env.settled()
    """

    def __init__(self, level: str = "o1_open_baseline", seed: int = 0):
        self.level_name = level
        self.spec = get(level)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.layout: dict = {}
        self.init_props: dict = {}
        self._ready = False          # 底座 __init__ 里会调用一次 reset()
        xml = (_HERE / self.spec["xml"]).resolve()
        super().__init__(xml_file=str(xml), seed=seed)

        # 道具 freejoint 映射（body 名 -> qpos 起始下标），通用不写死
        self.prop_adr: dict[str, int] = {}
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                bid = int(self.model.jnt_bodyid[j])
                self.prop_adr[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)] = \
                    int(self.model.jnt_qposadr[j])

        self._ready = True
        self.rng = np.random.default_rng(seed)
        self.reset()

    # ------------------------------------------------------------------
    def base_success(self) -> bool:
        """底座自带的判据：目标到圆盘的水平距离 <5cm、竖直在 (-3cm, 12cm)、夹爪张开。"""
        return bool(UranusGraspEnv.success(self))

    def prop_pos(self, body: str) -> np.ndarray:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)
        if i < 0:
            raise KeyError(f"场景里没有 body {body!r}（等级 {self.level_name}）")
        return self.data.xpos[i].copy()

    @property
    def instruction(self) -> str:
        return str(self.layout.get("instruction", ""))

    # ------------------------------------------------------------------
    def sample_layout(self, rng) -> dict:
        return self.spec["sample_layout"](rng)

    def _apply_layout(self, layout: dict) -> None:
        for name, pose in (layout.get("props") or {}).items():
            adr = self.prop_adr[name]
            self.data.qpos[adr:adr + 7] = np.asarray(pose, dtype=float)
        for name, pos in (layout.get("statics") or {}).items():
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0:
                raise KeyError(f"场景里没有 body {name!r}（等级 {self.level_name}）")
            self.model.body_pos[i] = np.asarray(pos, dtype=float)
        if layout.get("goal_xy") is not None:
            self.place_on_table(layout["goal_xy"])     # 底座实现：body_pos=[x,y,0.35+半高]
        mujoco.mj_forward(self.model, self.data)

    def reset(self, layout: dict | None = None, **kwargs):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        if not self._ready:                 # 底座 __init__ 里那次调用
            self.set_joints(self.q_home)
            self.set_gripper(open=True)
            return self.get_obs()

        self.layout = self.sample_layout(self.rng) if layout is None else dict(layout)
        if not self.layout.get("instruction"):     # 指令池抽一句（写进 dataset 的 task 字段）
            seen = bool(self.rng.random() < 0.75)
            pool = self.spec["instructions"] if seen else self.spec["holdout"]
            self.layout["instruction"] = str(self.rng.choice(pool))
            self.layout["instruction_split"] = "seen" if seen else "holdout"
        self._apply_layout(self.layout)
        self.set_joints(self.q_home)
        self.set_gripper(open=True)
        mujoco.mj_forward(self.model, self.data)
        self.init_props = {n: self.prop_pos(n) for n in (self.layout.get("props") or {})}
        return self.get_obs()

    # ------------------------------------------------------------------
    def success(self) -> bool:
        return bool(self.spec["success"](self))

    def settled(self) -> bool:
        return bool(self.spec["settled"](self))

    def metrics(self) -> dict:
        m = {
            "level": self.level_name,
            "title": self.spec["title"],
            "instruction": self.instruction,
            "obj_goal_dist": float(np.linalg.norm(self.get_object_pos() - self.get_place_target())),
            "obj_z": round(float(self.get_object_pos()[2]), 4),
            "success": self.success(),
            "settled": self.settled(),
        }
        for n in (self.layout.get("props") or {}):
            if n in self.init_props:
                m[f"{n}_moved_mm"] = round(float(
                    np.linalg.norm(self.prop_pos(n) - self.init_props[n])) * 1000.0, 1)
        return m

    def info(self) -> dict:
        key = {k: self.spec[k] for k in ("order", "title", "difficulty", "obstacle",
                                         "criterion", "xml", "hold_s") if k in self.spec}
        blob = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
        return {**key, "spec_hash": hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8],
                "seed": self.seed}


__all__ = ["ObstacleGraspEnv", "OBJ_HALF", "TABLE_Z"]

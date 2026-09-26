# -*- coding: utf-8 -*-
"""采样任务族的运行环境：`SamplingEnv`。

只做四件事，不提供任何专家策略（数据由人手操作采集）：
  1. 按等级加载场景（`scenes/l*.xml`）；
  2. **布局随机化**（`sample_layout` -> `_apply_layout`）；
  3. 每级的自动判据（`success` / `settled`）；
  4. 过程量（接触力、插入深度、里程碑）——给 HUD、dataset meta 和 selftest 用。

复用 `UranusGraspEnv` 的底座：齿面中点 6 自由度 IK、位置伺服残差闭环、夹爪宽度标定、
慢速释放。这些是拿血换来的，不重写。

命名约定（为了零改动复用底座）：场景里被操作物叫 `object`（带 `object_joint` freejoint 和
`object_geom`），目标点载体叫 `place_target`（带 `place_target_site`）——即使它实际是
采样篮 / 取样井 / 管壁立块 / 工具架，也沿用这套名字。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from .tasks import LEVELS, LEVEL_ORDER, SURFACE_Z, get

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent  # 仓库根（WYSD/）


def _find_demo_dir() -> Path | None:
    """定位 Lerobot-Uranus-VLA-Demo（优先复用 underwater_vision 的探测逻辑）。"""
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
except Exception as exc:  # pragma: no cover - 只在环境缺失时触发
    raise ImportError(
        f"找不到 Lerobot-Uranus-VLA-Demo（Uranus 底座所在仓库）：{exc}。"
        "把它的路径加进 PYTHONPATH，或设置环境变量 UNDERWATER_VISION_DEMO_DIR。"
    ) from exc


def _patch_numpy2_gripper() -> bool:
    """给 UranUS 底座打 numpy>=2 兼容补丁（`float(size-1 数组)` 在 numpy>=2 会 TypeError）。

    底座 `src/env.py::get_gripper` 写的是 `float(self.data.sensor(...).data.copy())`，
    在 numpy 1.x 下能跑、2.x 下抛 TypeError。**只在内存里 patch，不改他们的仓库文件**；
    已经修过就自动跳过。
    """
    try:
        float(np.asarray([1.0]))
        return False                     # numpy<2，不需要
    except TypeError:
        pass

    def _get_gripper(self):
        return float(np.asarray(self.data.sensor(self.GRIPPER_SENSOR).data).reshape(-1)[0])

    UranusGraspEnv.get_gripper = _get_gripper
    return True


_NUMPY2_PATCHED = _patch_numpy2_gripper()


class SamplingEnv(UranusGraspEnv):
    """一个等级的采样场景。用法：

        env = SamplingEnv("l1_rock_collect", seed=0)
        layout = env.sample_layout(env.rng)     # 或 env.reset() 内部自己抽
        env.reset(layout)
        ... 人手操作（或脚本下发 ctrl）...
        env.observe(dt); env.success(); env.settled()
    """

    def __init__(self, level: str = "l1_rock_collect", seed: int = 0):
        self.level_name = level
        self.spec = get(level)
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)
        self.layout: dict = {}
        self._ready = False          # 底座 __init__ 里会调用一次 reset()，那时还没准备好
        self._body_ids: dict[str, int] = {}
        self._site_ids: dict[str, int] = {}

        xml = (_HERE / self.spec["xml"]).resolve()   # 绝对路径，底座里 ROOT / abs 仍是 abs
        super().__init__(xml_file=str(xml), seed=seed)

        # ---- 道具 freejoint 映射：body 名 -> qpos 起始下标（通用，不写死任何等级）----
        self.prop_adr: dict[str, int] = {}
        self.prop_body_id: dict[str, int] = {}
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                bid = int(self.model.jnt_bodyid[j])
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, bid)
                self.prop_adr[name] = int(self.model.jnt_qposadr[j])
                self.prop_body_id[name] = bid

        # ---- 接触力用的 geom 集合 ----
        self.object_geoms = self._body_geoms(self.object_body_id)
        self.jaw_geoms = set()
        for b in range(self.model.nbody):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
            if n.startswith("jaw"):
                self.jaw_geoms |= self._body_geoms(b)
        tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "probe_tip_geom")
        self.tip_geom = int(tip) if tip >= 0 else None
        self.tip_site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "probe_tip_site"))
        self.tube_tip_site = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE,
                                                   "corer_tip_site"))

        self._ready = True
        self.rng = np.random.default_rng(seed)
        # 被操作物是铰接件（如 L8 阀门手柄）而不是自由体时，metrics 里多报一个关节角
        oj = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.OBJECT_JOINT)
        self.object_is_hinge = bool(
            oj >= 0 and self.model.jnt_type[oj] == mujoco.mjtJoint.mjJNT_HINGE)
        self.reset()

    # ------------------------------------------------------------------
    # 名字 -> id
    # ------------------------------------------------------------------
    def _body_geoms(self, body_id: int) -> set[int]:
        return {g for g in range(self.model.ngeom)
                if int(self.model.geom_bodyid[g]) == int(body_id)}

    def body_id(self, name: str) -> int:
        if name not in self._body_ids:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if i < 0:
                raise KeyError(f"场景里没有 body {name!r}（等级 {self.level_name}）")
            self._body_ids[name] = int(i)
        return self._body_ids[name]

    def site_id(self, name: str) -> int:
        if name not in self._site_ids:
            i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if i < 0:
                raise KeyError(f"场景里没有 site {name!r}（等级 {self.level_name}）")
            self._site_ids[name] = int(i)
        return self._site_ids[name]

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------
    def prop_pos(self, body: str) -> np.ndarray:
        return self.data.xpos[self.body_id(body)].copy()

    def site_pos(self, name: str) -> np.ndarray:
        return self.data.site_xpos[self.site_id(name)].copy()

    def probe_tip_pos(self) -> np.ndarray:
        """针尖位置：优先用场景里的 site（几何上最准，不用手算偏置）。"""
        if self.tip_site >= 0:
            return self.data.site_xpos[self.tip_site].copy()
        return self.get_object_pos()

    def object_tilt_deg(self) -> float:
        """被操作物的本体 z 轴相对世界 z 轴的夹角（度）。L5 用它判"管子竖不竖"。"""
        R = self.data.xmat[self.object_body_id].reshape(3, 3)
        c = float(np.clip(R[2, 2], -1.0, 1.0))
        return float(np.degrees(np.arccos(c)))

    def contact_force(self, a_geoms: set[int], b_geoms: set[int] | None) -> float:
        """接触力最大模值(N)。a 与 b 都给了就只算这两组之间的接触。"""
        best = 0.0
        buf = np.zeros(6, dtype=float)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 not in a_geoms and g2 not in a_geoms:
                continue
            if b_geoms is not None:
                if not ((g1 in a_geoms and g2 in b_geoms) or (g2 in a_geoms and g1 in b_geoms)):
                    continue
            mujoco.mj_contactForce(self.model, self.data, i, buf)
            best = max(best, float(np.linalg.norm(buf[:3])))
        return best

    @property
    def force_grasp(self) -> float:
        """夹爪对目标物的接触力（夹碎判据）。"""
        return self.contact_force(self.object_geoms, self.jaw_geoms)

    @property
    def force_tip(self) -> float:
        """针尖对外界的接触力（点测的"压过头"判据）。"""
        if self.tip_geom is None:
            return 0.0
        return self.contact_force({self.tip_geom}, None)

    @property
    def force_now(self) -> float:
        """当前等级真正关心的那个力。"""
        return self.force_tip if self.tip_geom is not None else self.force_grasp

    # ------------------------------------------------------------------
    # 布局随机化
    # ------------------------------------------------------------------
    def sample_layout(self, rng) -> dict:
        return self.spec["sample_layout"](rng)

    def place_on_table(self, xy=None, rng=None) -> np.ndarray:
        """把目标点载体摆到沉积物面上的 (x, y)。（覆盖底座：底座用的是桌高+物块半高）"""
        if xy is None:
            if rng is None:
                raise ValueError("place_on_table 需要 xy 或 rng 之一")
            xy = (rng.uniform(1.575, 1.625), rng.uniform(-0.16, -0.06))
        pos = np.array([float(xy[0]), float(xy[1]), SURFACE_Z])
        self.model.body_pos[self.place_body_id] = pos
        mujoco.mj_forward(self.model, self.data)
        return self.get_place_target()

    def _apply_layout(self, layout: dict) -> None:
        for name, pose in (layout.get("props") or {}).items():
            adr = self.prop_adr[name]
            self.data.qpos[adr:adr + 7] = np.asarray(pose, dtype=float)
        # 具名铰链/滑移关节（如 L8 的阀门开度）：直接写 qpos
        for name, val in (layout.get("joints") or {}).items():
            self.data.qpos[self.joint_adr(name)] = float(val)
        for name, pos in (layout.get("statics") or {}).items():
            self.model.body_pos[self.body_id(name)] = np.asarray(pos, dtype=float)
        if layout.get("goal_xy") is not None:
            gx, gy = layout["goal_xy"]
            self.model.body_pos[self.place_body_id] = np.array([gx, gy, SURFACE_Z])
        mujoco.mj_forward(self.model, self.data)

    def joint_adr(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"场景里没有 joint {name!r}（等级 {self.level_name}）")
        return int(self.model.jnt_qposadr[jid])

    def joint_angle(self, name: str = "object_joint") -> float:
        """读一个单自由度关节的角度（rad）。L8 用它当阀门开度。"""
        return float(self.data.qpos[self.joint_adr(name)])

    def joint_vel(self, name: str = "object_joint") -> float:
        """读一个单自由度关节的角速度（rad/s）。"""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"场景里没有 joint {name!r}")
        return float(self.data.qvel[int(self.model.jnt_dofadr[jid])])

    def reset(self, layout: dict | None = None, **kwargs):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        if not self._ready:                      # 底座 __init__ 里的那次调用
            self.set_joints(self.q_home)
            self.set_gripper(open=True)
            return self.get_obs()

        self.layout = self.sample_layout(self.rng) if layout is None else dict(layout)
        # 指令：等级自己的 layout 没给就按池子抽（75% 用训练模板，25% 用 hold-out）。
        # 抽到的这句会写进 LeRobotDataset 的 task 字段，所以同场景不同指令天然带语言-行为配对。
        if not self.layout.get("instruction"):
            seen = bool(self.rng.random() < 0.75)
            pool = self.spec["instructions"] if seen else self.spec["holdout"]
            self.layout["instruction"] = str(self.rng.choice(pool))
            self.layout["instruction_split"] = "seen" if seen else "holdout"
        self._apply_layout(self.layout)
        self.set_joints(self.q_home)
        self.set_gripper(open=True)
        mujoco.mj_forward(self.model, self.data)

        self.init_props = {n: self.prop_pos(n) for n in (self.layout.get("props") or {})}
        # 过程量清零
        self.peak_force = 0.0          # 夹持力峰值（L4 夹碎判据）
        self.peak_tip_force = 0.0
        self.max_depth = 0.0           # 管底相对沉积物面的最大下探深度
        self.milestone_lifted = False
        self.milestone_touch = False
        self._touch_hold = 0.0
        self.elapsed = 0.0
        self.observe(0.0)
        return self.get_obs()

    # ------------------------------------------------------------------
    # 过程量 / 判据
    # ------------------------------------------------------------------
    @property
    def instruction(self) -> str:
        return str(self.layout.get("instruction", ""))

    def observe(self, dt: float = 0.05) -> None:
        """每个控制拍调用一次（采集器/脚本都要调）。累计峰值力与里程碑。"""
        self.elapsed += float(dt)
        fg, ft = self.force_grasp, self.force_tip
        self.peak_force = max(self.peak_force, fg)
        self.peak_tip_force = max(self.peak_tip_force, ft)
        self.force_grasp_now, self.force_tip_now = fg, ft

        # 管底下探深度（L5 过程量）：优先用管底 site（几何最准），否则退回管心 - 半径
        if self.tube_tip_site >= 0:
            depth = SURFACE_Z - float(self.data.site_xpos[self.tube_tip_site][2])
            self.max_depth = max(self.max_depth, depth)
        elif "object" in self.prop_adr:
            depth = SURFACE_Z + self.object_half - float(self.get_object_pos()[2])
            self.max_depth = max(self.max_depth, depth)

        # 里程碑（L7）：① 被拿起来过 ② 点测保持达标
        if self.init_props and "object" in self.init_props:
            if float(self.get_object_pos()[2] - self.init_props["object"][2]) > 0.03:
                self.milestone_lifted = True
        tol = (self.spec.get("tol") or {}).get("tip")
        if tol is not None:
            d = float(np.linalg.norm(self.probe_tip_pos() - self.get_place_target()))
            if d < tol and ft < self.spec.get("force_limit", 1e9):
                self._touch_hold += float(dt)
            else:
                self._touch_hold = 0.0
            if self._touch_hold >= 0.3:
                self.milestone_touch = True

    def success(self) -> bool:
        return bool(self.spec["success"](self))

    def settled(self) -> bool:
        fn = self.spec.get("settled")
        return True if fn is None else bool(fn(self))

    def metrics(self) -> dict:
        m = {
            "level": self.level_name,
            "title": self.spec["title"],
            "instruction": self.instruction,
            "obj_goal_dist": float(np.linalg.norm(self.get_object_pos() - self.get_place_target())),
            "tilt_deg": self.object_tilt_deg(),
            "force_grasp": round(float(self.force_grasp), 2),
            "force_tip": round(float(self.force_tip), 2),
            "peak_force": round(float(self.peak_force), 2),
            "max_depth_mm": round(float(self.max_depth) * 1000.0, 1),
            "success": self.success(),
            "settled": self.settled(),
            "milestone_lifted": bool(self.milestone_lifted),
            "milestone_touch": bool(self.milestone_touch),
            "elapsed_s": round(float(self.elapsed), 2),
        }
        if self.tip_site >= 0:
            m["tip_site_dist_mm"] = round(
                float(np.linalg.norm(self.probe_tip_pos() - self.get_place_target())) * 1000.0, 1)
        if self.object_is_hinge:
            m["valve_deg"] = round(float(np.degrees(self.joint_angle())), 1)
            m["valve_target_deg"] = round(
                float(np.degrees(self.layout.get("target_angle", 0.0))), 1)
            m["valve_start_deg"] = round(
                float(np.degrees(self.layout.get("start_angle", 0.0))), 1)
        return m

    def info(self) -> dict:
        """写进 dataset meta 的规格摘要（保证"同一次采集用的是哪一级、什么指令、什么水况"可追溯）。"""
        key = {k: self.spec[k] for k in
               ("order", "title", "difficulty", "xml", "hold_s", "water",
                "force_limit", "tol", "object_desc", "goal_desc")
               if k in self.spec}
        blob = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
        return {**key, "spec_hash": hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8],
                "seed": self.seed}

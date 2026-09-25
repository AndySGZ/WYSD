"""Uranus 点到点到达(reach)环境的精简实现。

与教程 `Lerobot-MujoCo-VLA-Tutorial/src/env/env.py` 对应，但做了关键简化/适配：

1. 用原生 mujoco 实现（不依赖教程庞大的 MuJoCoParserClass + glfw viewer），
   因此可以完全无头(headless)运行，便于脚本化采集 / 训练 / 评测。
2. 真实 Uranus 是"并联传动 + 腱驱动夹爪"：
   - 物理臂有 6 个串联关节 joint1..joint6；
   - 真实机器上 joint2/joint3 由液压缸(cylinder1/2_slide)并联驱动，需要
     `CylinderMapper`(uranus/control/cylinder_mapper.py) 做"关节角->缸体滑动量"映射。
   - 但该并联机构在 mj_step 动力学下数值极不稳定，无法稳定跟踪关节目标。
     为了让本 demo 能用真实接触动力学做抓取，模型里把 joint2/joint3 改成直接
     位置执行器（串联简化，见 asset/uranus/uranus_arm_gripper_model.xml 的注释），
     缸体本身保留为被动机构仅作可视化。
   因此本环境的 `step_dynamics()` 使用"6 关节 + 1 夹爪"的直接 ctrl 映射；
   `joints_to_ctrl_hydraulic()` 仍保留真实液压映射，供参考/对比。
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent.parent


class UranusReachEnv:
    """Uranus 大臂点对点到达任务。

    obs  = [joint1..joint6 (当前, 6), goal_x, goal_y, goal_z (3)]   -> 9 维
    action = [joint1..joint6 (目标, 6), gripper(1)]                 -> 7 维（gripper 本任务恒开）
    """

    ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    GRIPPER_JOINTS = ["jaw1.1", "jaw2.1"]
    TCP_BODY = "tcp_link"
    GRIPPER_SENSOR = "gripper_position"
    # 夹爪开/合对应的 2 个基座关节角（jaw1.1, jaw2.1）
    # 标定（2026-09-11 实测，见 grasp_gap 标定）：
    #   齿面中心距(m) ≈ 0.0231 + 0.246 * jaw_angle   （jaw_angle ∈ [-0.25, 0.35]）
    #   齿面净间距 = 齿面中心距 - 0.0102（齿厚 2*0.0051）
    # 于是：0.22 rad -> 净间距 66.7mm（可容下 <=6cm 的物块），
    #       0.00 rad -> 净间距 12.9mm，-0.07 rad -> 两齿面刚好贴住。
    GRIPPER_OPEN = np.array([0.22, 0.22])
    GRIPPER_CLOSE = np.array([-0.07, -0.07])
    GRIPPER_OPEN_ANGLE = 0.22  # 张开时 jaw1.1 关节角（get_gripper 的返回值）

    def __init__(
        self,
        xml_file: str = "asset/scene_uranus.xml",
        goal_r=(1.2, 1.75),
        goal_theta_deg=(-80.0, 80.0),
        goal_z=(0.20, 0.90),
        success_thresh: float = 0.15,
        seed: int = 0,
        reset_on_init: bool = True,
    ):
        xml_path = ROOT / xml_file
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # 6 个串联关节的 qpos 索引与限位
        self.arm_qpos_idx = np.array(
            [self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in self.ARM_JOINTS]
        )
        self.arm_dofadr = np.array(
            [self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]
             for j in self.ARM_JOINTS]
        )
        self.q_min = np.array([self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j), 0]
                               for j in self.ARM_JOINTS])
        self.q_max = np.array([self.model.jnt_range[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j), 1]
                               for j in self.ARM_JOINTS])

        self.tcp_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.TCP_BODY)
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")

        # 注：本 demo 已将液压缸体从模型中移除（纯串联简化），
        # CylinderMapper 仅保留在 uranus/control/ 下作为历史参考，不再使用。

        self.goal_r = goal_r
        self.goal_theta_deg = goal_theta_deg
        self.goal_z = goal_z
        self.success_thresh = success_thresh
        self.rng = np.random.default_rng(seed)

        # 先复位到 home 并读取 home 关节（sample_goal/solve_ik 依赖 self.q_home）
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        self.q_home = self.get_joints().copy()

        # 再正式 reset（会采样目标）；子类可在设置完自身属性后再手动 reset
        if reset_on_init:
            self.reset()

    # ------------------------------------------------------------------
    # 状态读取 / 写入
    # ------------------------------------------------------------------
    def get_joints(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos_idx].copy()

    def set_joints(self, q: np.ndarray):
        q = np.clip(np.asarray(q, dtype=float), self.q_min, self.q_max)
        self.data.qpos[self.arm_qpos_idx] = q
        mujoco.mj_forward(self.model, self.data)

    def get_tcp_pos(self) -> np.ndarray:
        return self.data.xpos[self.tcp_body_id].copy()

    def get_gripper(self) -> float:
        return float(self.data.sensor(self.GRIPPER_SENSOR).data.copy())

    def get_obs(self) -> np.ndarray:
        return np.concatenate([self.get_joints(), self.goal]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return len(self.ARM_JOINTS) + 3

    @property
    def action_dim(self) -> int:
        return len(self.ARM_JOINTS) + 1

    # ------------------------------------------------------------------
    # 复位 / 目标
    # ------------------------------------------------------------------
    def reset(self, goal: np.ndarray | None = None):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        if goal is None:
            goal = self.sample_goal()
        self.set_goal(goal)
        # sample_goal 内部的 IK 会把臂移动到目标构型，这里复位回 home，
        # 保证每个 episode 都从 home 构型出发（而非已经到达目标）。
        self.set_joints(self.q_home)
        return self.get_obs()

    def set_goal(self, goal: np.ndarray):
        self.goal = np.asarray(goal, dtype=float)
        # 把场景里的目标块挪到 goal 位置
        self.model.body_pos[self.target_body_id] = self.goal
        mujoco.mj_forward(self.model, self.data)

    def sample_goal(self) -> np.ndarray:
        lo_r, hi_r = self.goal_r
        lo_th, hi_th = self.goal_theta_deg
        lo_z, hi_z = self.goal_z
        for _ in range(200):
            r = self.rng.uniform(lo_r, hi_r)
            th = np.deg2rad(self.rng.uniform(lo_th, hi_th))
            z = self.rng.uniform(lo_z, hi_z)
            goal = np.array([r * np.cos(th), r * np.sin(th), z])
            q, err = self.solve_ik(goal)
            if np.linalg.norm(err) < 0.02:  # IK 可解，说明可达
                return goal
        # 兜底：直接用 home 附近的点
        return np.array([1.6, 0.0, 0.46])

    # ------------------------------------------------------------------
    # 数值 IK（在 6 个串联关节上，位置级）
    # ------------------------------------------------------------------
    def solve_ik(self, target_pos: np.ndarray, q0=None, max_iter=400, tol=1e-3):
        if q0 is None:
            q0 = self.q_home
        q = np.asarray(q0, dtype=float).copy()
        J = np.zeros((3, self.model.nv))
        for _ in range(max_iter):
            self.set_joints(q)
            err = target_pos - self.get_tcp_pos()
            if np.linalg.norm(err) < tol:
                break
            mujoco.mj_jacBody(self.model, self.data, J, None, self.tcp_body_id)
            J6 = J[:, self.arm_dofadr]           # (3, 6)
            dq = np.linalg.lstsq(J6, err, rcond=1e-3)[0]
            dq = np.clip(dq, -0.3, 0.3)
            q = np.clip(q + dq, self.q_min, self.q_max)
        self.set_joints(q)
        return q, target_pos - self.get_tcp_pos()

    # ------------------------------------------------------------------
    # 交互接口（与教程风格一致）
    # ------------------------------------------------------------------
    def step(self, action: np.ndarray):
        """运动学 step：直接把 action 前 6 维当作目标关节角。"""
        q_target = np.asarray(action[:6], dtype=float)
        self.set_joints(q_target)
        obs = self.get_obs()
        dist = float(np.linalg.norm(self.get_tcp_pos() - self.goal))
        done = dist < self.success_thresh
        info = {"tcp": self.get_tcp_pos(), "goal": self.goal, "dist": dist, "success": done}
        return obs, info

    def joints_to_ctrl_hydraulic(self, q6: np.ndarray, gripper_cmd: float = 0.0) -> np.ndarray:
        """真实 Uranus 的"关节角 -> 液压缸 ctrl"映射。

        本 demo 已把液压缸体从模型中移除（纯串联简化），因此该映射不再适用，
        仅保留历史接口。需要时可参考 uranus/control/cylinder_mapper.py。
        """
        raise NotImplementedError(
            "液压缸已从模型中移除；请参考 uranus/control/cylinder_mapper.py 理解原始映射。"
        )

    def step_dynamics(self, action: np.ndarray, substeps: int = 5):
        """动力学 step：把 action 前 6 维当作目标关节角，直接下发 data.ctrl 再 mj_step。

        使用串联简化（模型里 joint2/joint3 已改为直接位置执行器），夹爪为 2 个
        基座关节直接位置执行器，因此 ctrl = [joint1..joint6, jaw1.1, jaw2.1]。
        """
        q_target = np.clip(np.asarray(action[:6], dtype=float), self.q_min, self.q_max)
        gripper = float(action[-1])
        gripper_angles = self.GRIPPER_CLOSE if gripper > 0.5 else self.GRIPPER_OPEN
        ctrl = np.concatenate([q_target, gripper_angles])
        self.data.ctrl[:] = ctrl
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        obs = self.get_obs()
        dist = float(np.linalg.norm(self.get_tcp_pos() - self.goal))
        info = {"tcp": self.get_tcp_pos(), "goal": self.goal, "dist": dist, "ctrl": ctrl}
        return obs, info

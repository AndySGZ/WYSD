"""Uranus 抓取(pick-and-place)环境：在 UranusReachEnv 基础上加桌子、物块、放置目标。

物块是桌面上的一个立方体（freejoint，尺寸读自 scene xml，当前 5cm）。任务流程：
approach(开爪) -> 闭环下探到物块中心 -> 按物块宽度闭爪 -> 抬升 -> 移到放置点
-> 下降 -> 缓慢张开释放。

动力学用串联简化（见 env.py 注释）；抓取对齐用"齿面中点"（两片齿 geom 的中点）而不是
TCP，因为齿面接触点相对 TCP 有固定偏置。几个关键实现细节（都是实测踩出来的）：

- `solve_ik_grasp`：6 自由度 IK（位置 + 姿态），姿态带 10° 下俯（GRASP_PITCH），
  否则夹爪本体会顶在桌面上；默认只求解、不改仿真状态，避免"夹着物块时把手臂瞬移走"。
- `move_to`：位置伺服有 ~1cm 重力垂沉，用实测残差闭环修正 IK 目标。
- `align_jaws`：按物块宽度给两片齿做平行补偿（齿面法向随 jaw 角转 2a）。
- `release`：缓慢张开，直接跳到张开角会把物块弹飞。
"""

from __future__ import annotations

import mujoco
import numpy as np

from src.env import UranusReachEnv


class UranusGraspEnv(UranusReachEnv):
    OBJECT_BODY = "object"
    OBJECT_JOINT = "object_joint"
    OBJECT_GEOM = "object_geom"
    PLACE_BODY = "place_target"
    PLACE_SITE = "place_target_site"
    TOOTH1_GEOM = "jaw1_tooth_col"
    TOOTH2_GEOM = "jaw2_tooth_col"
    JAW1_BODY = "jaw1.1.1"
    JAW2_BODY = "jaw2.1.1"

    # 抓取姿态 + 夹爪标定见下方常量；齿面中点 IK 由 solve_ik_grasp 做 6 自由度求解。

    # ---- 夹爪标定（实测：齿面中心距 ≈ GAP0 + GAP_K * jaw_angle）----
    # 单位 m / rad。用来按"物块宽度"反解闭合角度，这样换物体尺寸不用改代码。
    GRIPPER_GAP0 = 0.0231   # jaw_angle = 0 时两齿面中心的间距
    GRIPPER_GAP_K = 0.246   # 每弧度张开的间距增量
    TOOTH_HALF_THICK = 0.0051   # 齿面沿闭合方向(y)的半厚
    TOOTH_HALF_LONG = 0.0364    # 齿面沿长轴(x)的半长

    # 抓取姿态：齿条长轴相对水平面下俯的角度。
    # 0° 时夹爪本体(link6_mount, 半径 3cm)比齿面中点低 3cm，会扎进桌面(z=0.35)，
    # 手臂下不到物块中心（实测停在物块中心上方 1cm 处顶住）；
    # 10° 可把腕部抬到桌面上方，齿的前下角也仍有 ~6mm 余量。
    GRASP_PITCH = np.deg2rad(10.0)

    # 随机化范围（台面 x∈[1.52,1.92]、y∈[-0.24,0.20]，都留出物块半径 + 1cm 余量）
    # x 上限受"夹爪本体不能让开台面近边"约束：齿面中点在 x 时，link6_mount 会扫到
    # x-0.136 且低于桌面，所以 x 必须 < 台面近边(1.52)+0.136≈1.656。
    OBJECT_X_RANGE = (1.555, 1.645)
    OBJECT_Y_RANGE = (-0.10, 0.12)
    PLACE_X_RANGE = (1.555, 1.645)
    PLACE_Y_RANGE = (-0.18, 0.04)
    MIN_PICK_PLACE_DIST = 0.10   # 抓取点与放置点的最小平面距离

    def __init__(self, xml_file="asset/scene_uranus_grasp.xml", seed=0, **kwargs):
        # 抓取场景目标采样不需要 reach 的极坐标目标，直接关掉相关默认值
        kwargs.setdefault("goal_r", (1.2, 1.75))
        kwargs.setdefault("goal_theta_deg", (-80.0, 80.0))
        kwargs.setdefault("goal_z", (0.2, 0.9))
        super().__init__(xml_file=xml_file, seed=seed, reset_on_init=False, **kwargs)

        self.object_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.OBJECT_BODY)
        self.object_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.OBJECT_GEOM)
        self.object_jnt_qposadr = self.model.jnt_qposadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.OBJECT_JOINT)
        ]
        self.place_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.PLACE_SITE)
        self.place_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.PLACE_BODY)
        self.tooth1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.TOOTH1_GEOM)
        self.tooth2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, self.TOOTH2_GEOM)
        self.jaw1_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.JAW1_BODY)
        self.jaw2_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.JAW2_BODY)

        # 物块尺寸直接从 XML 的 geom size 读，换物体大小只需改 scene xml
        self.object_half = float(self.model.geom_size[self.object_geom_id][0])
        self.object_width = 2.0 * self.object_half

        # 夹爪关节角范围（用于把反解出的闭合角 clip 进可行域）
        jaw_jnt = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, self.GRIPPER_JOINTS[0])
        self.jaw_min, self.jaw_max = (float(v) for v in self.model.jnt_range[jaw_jnt])

        # 按当前物块尺寸给齿面做平行补偿 + 配张开角（换物块尺寸后调用 align_jaws() 即可）
        self.open_angle = float(self.GRIPPER_OPEN[0])
        self.align_jaws()

        # 物块初始位姿（读取 home keyframe 里 freejoint 的前 7 维）
        self.object_home = self.data.qpos[self.object_jnt_qposadr: self.object_jnt_qposadr + 7].copy()

        # 正式 reset（臂回 home + 夹爪打开）
        self.reset()

    # ------------------------------------------------------------------
    # 物块 / 夹爪状态
    # ------------------------------------------------------------------
    def get_object_pos(self) -> np.ndarray:
        return self.data.xpos[self.object_body_id].copy()

    def get_object_quat(self) -> np.ndarray:
        return self.data.xquat[self.object_body_id].copy()

    def get_object_pose(self) -> np.ndarray:
        """物块位姿：xyz(3) + 四元数(4) = 7 维。"""
        return np.concatenate([self.get_object_pos(), self.get_object_quat()]).astype(np.float32)

    def get_place_target(self) -> np.ndarray:
        return self.data.site_xpos[self.place_site_id].copy()

    def place_on_table(self, xy=None, rng=None) -> np.ndarray:
        """把放置目标(视觉标记 + success 判定点)放到桌面上的 (x,y) 处，z 取物块中心高度。

        传 rng 时按 PLACE_X/Y_RANGE 随机采样。返回放置点 xyz。
        """
        if xy is None:
            if rng is None:
                raise ValueError("place_on_table 需要 xy 或 rng 之一")
            xy = (rng.uniform(*self.PLACE_X_RANGE), rng.uniform(*self.PLACE_Y_RANGE))
        pos = np.array([float(xy[0]), float(xy[1]), 0.35 + self.object_half])
        self.model.body_pos[self.place_body_id] = pos
        mujoco.mj_forward(self.model, self.data)
        return pos

    def sample_layout(self, rng) -> tuple[np.ndarray, np.ndarray]:
        """随机一个"物块初始位姿 + 放置点"，两者在台面上至少隔开 MIN_PICK_PLACE_DIST。

        物块返回 7 维位姿(xyz + 单位四元数)，直接喂给 reset(object_pose=...)。
        """
        for _ in range(200):
            oxy = np.array([rng.uniform(*self.OBJECT_X_RANGE), rng.uniform(*self.OBJECT_Y_RANGE)])
            pxy = np.array([rng.uniform(*self.PLACE_X_RANGE), rng.uniform(*self.PLACE_Y_RANGE)])
            if np.linalg.norm(oxy - pxy) < self.MIN_PICK_PLACE_DIST:
                continue
            obj_pose = np.array([oxy[0], oxy[1], 0.35 + self.object_half, 1.0, 0.0, 0.0, 0.0])
            self.place_on_table(pxy)
            return obj_pose, self.get_place_target().copy()
        raise RuntimeError("采样不到合法的物块/放置点组合")

    def set_object_xy(self, xy) -> np.ndarray:
        """把物块摆到桌面上指定 (x,y)，z 取"桌面 + 半高"。返回 7 维位姿。"""
        pose = np.array([float(xy[0]), float(xy[1]), 0.35 + self.object_half, 1.0, 0.0, 0.0, 0.0])
        self.set_object_pose(pose)
        return pose

    def tooth_midpoint(self) -> np.ndarray:
        return (self.data.geom_xpos[self.tooth1_id] + self.data.geom_xpos[self.tooth2_id]) / 2.0

    def tooth_gap(self) -> float:
        return float(np.linalg.norm(
            self.data.geom_xpos[self.tooth1_id] - self.data.geom_xpos[self.tooth2_id]
        ))

    # ------------------------------------------------------------------
    # 夹爪角度 <-> 物块宽度
    # ------------------------------------------------------------------
    def tooth_clearance(self, jaw_angle: float) -> float:
        """两齿面之间的净间距(m)。

        注意：只有在"齿面平行"的张开度（也就是 align_jaws 用的 grasp_angle）下这个式子
        才严格成立 —— 6 自由度 IK 会把 tooth1 的坐标系钉到世界系，所以齿面在世界系里
        始终是正对的；张开度偏离标称值时两齿面才出现 2*(a-a*) 的相对倾斜。
        """
        return self.GRIPPER_GAP0 + self.GRIPPER_GAP_K * float(jaw_angle) - 2.0 * self.TOOTH_HALF_THICK

    def measured_clearance(self) -> float:
        """当前仿真状态下两齿面之间的实际净间距（按几何投影算，用于校验/报告）。"""
        def proj_half(gid, axis):
            m = self.data.geom_xmat[gid].reshape(3, 3)
            return (abs(m[:, 0] @ axis) * self.TOOTH_HALF_LONG
                    + abs(m[:, 1] @ axis) * self.TOOTH_HALF_THICK
                    + abs(m[:, 2] @ axis) * 0.008)
        d = self.data.geom_xpos[self.tooth2_id] - self.data.geom_xpos[self.tooth1_id]
        n = d / max(float(np.linalg.norm(d)), 1e-9)
        return float(np.linalg.norm(d) - proj_half(self.tooth1_id, n) - proj_half(self.tooth2_id, n))

    def _angle_for_clearance(self, target: float) -> float:
        a = (target + 2.0 * self.TOOTH_HALF_THICK - self.GRIPPER_GAP0) / self.GRIPPER_GAP_K
        return float(np.clip(a, self.jaw_min, self.jaw_max))

    def grasp_angle(self, width: float | None = None, squeeze: float = 0.004) -> float:
        """夹住宽度 width 的物块所需的夹爪角。

        令齿面净间距 = width - squeeze（squeeze 是压入量，产生夹持力）。
        width 默认取场景里物块的实际宽度，所以换物块尺寸无需改代码。
        """
        w = self.object_width if width is None else float(width)
        return self._angle_for_clearance(w - squeeze)

    def open_angle_for(self, width: float | None = None, margin: float = 0.012) -> float:
        """张开到"物块宽度 + margin"所需的角度（齿面做了平行补偿后会变斜，开度要按此重算）。"""
        w = self.object_width if width is None else float(width)
        return self._angle_for_clearance(w + margin)

    def align_jaws(self, width: float | None = None, open_margin: float = 0.014):
        """按目标物块宽度给两片齿做"平行补偿"，并配好对应的张开角。

        齿面法向随 jaw 角转动约 2a：张得越开两齿面越不平行，闭合时会把方块沿齿面
        挤出去（物块越大越严重，实测 5cm 方块直接被挤出 9cm）。给两片齿各预旋 ∓a*
        （a* 就是夹该物块所需的张开角），就能在夹持张开度下让两齿面严格平行、
        且垂直于闭合方向 —— 等价于换上"为这个物块配对"的齿形垫。
        张开角按 width + open_margin 一并配好，所以换物块尺寸只需改 scene xml 的 geom size。
        """
        w = self.object_width if width is None else float(width)
        a = self.grasp_angle(w)
        self.model.geom_quat[self.tooth1_id] = np.array([np.cos(-a / 2), 0.0, 0.0, np.sin(-a / 2)])
        self.model.geom_quat[self.tooth2_id] = np.array([np.cos(+a / 2), 0.0, 0.0, np.sin(+a / 2)])
        self.open_angle = float(np.clip(self.open_angle_for(w, open_margin), a, self.jaw_max))
        mujoco.mj_forward(self.model, self.data)
        return a

    def _grip_ctrl(self, open: bool, squeeze: float = 0.004) -> np.ndarray:
        """张/合对应的 ctrl（合是按物块宽度标定过的闭合角，而不是"夹到底"）。"""
        if open:
            return np.full(2, self.open_angle)
        return np.full(2, self.grasp_angle(squeeze=squeeze))

    def reset(self, object_pose: np.ndarray | None = None, **kwargs):
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key_id)
        mujoco.mj_forward(self.model, self.data)
        if object_pose is not None:
            self.set_object_pose(object_pose)
        # 臂回 home，夹爪打开
        self.set_joints(self.q_home)
        self.set_gripper(open=True)
        return self.get_obs()

    def set_object_pose(self, pose: np.ndarray):
        self.data.qpos[self.object_jnt_qposadr: self.object_jnt_qposadr + 7] = np.asarray(pose, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, open: bool = True, squeeze: float = 0.004):
        """把夹爪 2 个基座关节设到开/合对应角度（写 ctrl）。"""
        angles = self._grip_ctrl(open, squeeze)
        # 写 ctrl（动力学执行器目标）
        gripper_act_idx = np.array([
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j + "_act")
            for j in self.GRIPPER_JOINTS
        ])
        for act, ang in zip(gripper_act_idx, angles):
            self.data.ctrl[act] = ang

    # ------------------------------------------------------------------
    # 齿面中点 IK（抓取对齐用）
    # ------------------------------------------------------------------
    @staticmethod
    def _rotvec(R: np.ndarray) -> np.ndarray:
        """把旋转矩阵转成轴角(旋转向量)；接近单位阵时返回 0。"""
        c = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        th = np.arccos(c)
        if th < 1e-10:
            return np.zeros(3)
        v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
        return th * v

    def solve_ik_grasp(self, target_pos: np.ndarray, q0=None, max_iter=500, tol=1e-3,
                       pitch: float | None = None, restore_state: bool = True):
        """把"齿面中点"移动到 target_pos，同时把夹爪姿态摆正的 6 自由度 IK。

        只用位置 IK(3 DOF)时，齿条的长轴会随腕部俯仰倾斜 ~28°，导致齿的端面而不是
        平面对上物块、夹不住。这里同时约束：
          - 位置：齿面中点 -> target_pos（3 DOF）
          - 姿态：齿的长轴沿 x 并下俯 pitch、齿面法向沿 y、高度轴沿 z（3 DOF）
        用 6x6 雅可比求解。

        pitch（默认 GRASP_PITCH=10°，即齿条长轴下俯 10°）：夹爪本体 link6_mount 比齿面
        中点低约 3cm，纯水平姿态下会扎进桌面(z=0.35)，手臂根本下不到物块中心；下俯一点
        能把整个腕部抬起来让开桌面，而且更接近真实抓取姿态（齿面依然正对物块 ±y 面）。

        restore_state=True 时只做"求解"，不改动仿真状态（IK 内部要改关节角算雅可比，
        若不复原，夹着物块时这一步等于把手臂瞬移走、物块就被丢在原地了）。
        """
        if pitch is None:
            pitch = self.GRASP_PITCH
        if q0 is None:
            q0 = self.q_home
        q = np.asarray(q0, dtype=float).copy()
        saved_qpos = self.data.qpos.copy() if restore_state else None
        c, s = float(np.cos(pitch)), float(np.sin(pitch))
        R_des = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        jacp1 = np.zeros((3, self.model.nv))
        jacp2 = np.zeros((3, self.model.nv))
        jacp_b = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        J = np.zeros((6, 6))
        for _ in range(max_iter):
            self.set_joints(q)
            epos = np.asarray(target_pos, dtype=float) - self.tooth_midpoint()
            R_cur = self.data.geom_xmat[self.tooth1_id].reshape(3, 3)
            eori = self._rotvec(R_des @ R_cur.T)
            err = np.concatenate([epos, eori])
            if np.linalg.norm(epos) < tol and np.linalg.norm(eori) < 0.01:
                break
            mujoco.mj_jac(self.model, self.data, jacp1, None,
                          self.data.geom_xpos[self.tooth1_id], self.jaw1_body_id)
            mujoco.mj_jac(self.model, self.data, jacp2, None,
                          self.data.geom_xpos[self.tooth2_id], self.jaw2_body_id)
            mujoco.mj_jacBody(self.model, self.data, jacp_b, jacr, self.tcp_body_id)
            J[:3] = 0.5 * (jacp1 + jacp2)[:, self.arm_dofadr]
            J[3:] = jacr[:, self.arm_dofadr]
            dq = np.linalg.lstsq(J, err, rcond=1e-3)[0]
            dq = np.clip(dq, -0.2, 0.2)
            q = np.clip(q + dq, self.q_min, self.q_max)
        self.set_joints(q)
        err = np.asarray(target_pos, dtype=float) - self.tooth_midpoint()
        if saved_qpos is not None:
            self.data.qpos[:] = saved_qpos
            mujoco.mj_forward(self.model, self.data)
        return q, err

    # ------------------------------------------------------------------
    # 动力学 move：下发放目标关节角 + 夹爪开关，跑 substeps 步
    # ------------------------------------------------------------------
    def move(self, q_target: np.ndarray, gripper_open: bool, substeps: int = 300,
             squeeze: float = 0.004, hook=None):
        """下发 [目标关节角, 夹爪] 并跑 substeps 步动力学。

        hook(i, q_cmd, jaw_cmd)：每步 mj_step 之后回调，供数据采集按控制周期录制
        (obs, action)；jaw_cmd 是实际下发的夹爪关节角（张开角/夹持角）。
        """
        q = np.clip(np.asarray(q_target, dtype=float), self.q_min, self.q_max)
        grip = self._grip_ctrl(gripper_open, squeeze)
        ctrl = np.concatenate([q, grip])
        self.data.ctrl[:] = ctrl
        for i in range(substeps):
            mujoco.mj_step(self.model, self.data)
            if hook is not None:
                hook(i, q, float(grip[0]))
        return self.get_obs()

    def move_to(self, target_pos: np.ndarray, gripper_open: bool, substeps: int = 800,
                iters: int = 4, tol: float = 0.002, gain: float = 1.0,
                squeeze: float = 0.004, q0: np.ndarray | None = None, hook=None):
        """闭环把"齿面中点"开到 target_pos（位置级）。

        位置执行器在重力下会有静态垂沉（实测大臂伸直时约 1cm），单纯"IK 一次 + mj_step"
        到不了位；这里用实测齿面中点残差反过来修正 IK 目标，迭代几次即可收敛到 mm 级。
        """
        tgt = np.asarray(target_pos, dtype=float).copy()
        q = None
        for _ in range(iters):
            q, _ = self.solve_ik_grasp(tgt, q0=self.get_joints() if q0 is None else q0)
            self.move(q, gripper_open, substeps, squeeze=squeeze, hook=hook)
            resid = tgt - self.tooth_midpoint()
            if np.linalg.norm(resid) < tol:
                break
            tgt = tgt + gain * resid
        return q, tgt - self.tooth_midpoint()

    def release(self, q_target: np.ndarray | None = None, substeps: int = 1500, hook=None):
        """原地缓慢张开夹爪放下物块。

        直接把 ctrl 从夹持角跳到张开角会猛地把物块弹飞（实测被推出桌外、水平误差 17cm）；
        按 ~3s 线性张开则物块几乎原地落下（水平误差 ~1.5cm）。
        夹爪角是连续动作，所以这段斜坡会被采集进 action，策略能学到"慢慢张开"。
        """
        q = (self.get_joints() if q_target is None
             else np.clip(np.asarray(q_target, dtype=float), self.q_min, self.q_max))
        a0 = float(self.get_gripper())
        a1 = float(self.open_angle)
        for i in range(substeps):
            a = a0 + (a1 - a0) * (i + 1) / substeps
            self.data.ctrl[:] = np.concatenate([q, [a, a]])
            mujoco.mj_step(self.model, self.data)
            if hook is not None:
                hook(i, q, float(a))
        return self.get_obs()

    def get_obs(self) -> np.ndarray:
        """obs = [关节(6), 物块相对齿面中点(3), 物块四元数(4), 放置点相对齿面中点(3), 夹爪(1)] = 17。

        位置量都用"相对齿面中点"而不是世界坐标：策略要学的是"物块在我手的哪个方位、
        该往哪走"，相对表示对物块摆放位置不敏感，BC 泛化明显更好（实测绝对坐标版本
        在随机布局下闭环误差累积很快）。
        """
        tm = self.tooth_midpoint()
        return np.concatenate([
            self.get_joints(),
            self.get_object_pos() - tm,
            self.get_object_quat(),
            self.get_place_target() - tm,
            [self.get_gripper()],
        ]).astype(np.float32)

    @property
    def obs_dim(self) -> int:
        return 6 + 3 + 4 + 3 + 1

    def success(self, pos_tol: float = 0.05, gripper_open_tol: float = 0.1) -> bool:
        """物块到达放置点正上方且夹爪已张开。"""
        obj = self.get_object_pos()
        target = self.get_place_target()
        horizontal = np.linalg.norm(obj[:2] - target[:2])
        vertical = obj[2] - target[2]
        gripper = self.get_gripper()
        return (horizontal < pos_tol) and (-0.03 < vertical < 0.12) and (gripper > self.open_angle - gripper_open_tol)

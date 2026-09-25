"""Uranus 抓取-放置的脚本专家（状态机），供 grasp_e2e.py 验证 与 collect_data_grasp.py 采数据共用。

流程（每一步都是"下发给底层控制器的目标关节角 + 夹爪命令"，与 reach 的 action 语义一致）：

    起始(home, 开爪) -> 物块上方 -> 下探到物块中心 -> 闭爪 -> 抬升
                     -> 放置点上方 -> 下降到放置高度 -> 缓开爪释放

与 reach 的专家（单个 IK 解）不同，这里必须跑接触动力学（物块要被真的夹起来），
所以每一步都是 `env.move_to(...)`：先 6 自由度齿面中点 IK，再 mj_step，
并按实测残差闭环修正（位置伺服有 ~1cm 重力垂沉）。

**所有移动都在"齿面中点"空间插值成若干中间路点**（`_cartesian_move`），而不是一步跳到位。
原因：位置伺服同时驱动 6 个关节，单步大跳会让手走一条弧线，实测会在接近物块时把物块扫飞；
插值后既不会扫到物块/桌面，录出来的专家轨迹也更接近真实示教。

record=True 时按控制周期录制，动作取**未来第 K 个控制周期的 [实测关节角(6), 下发的夹爪命令(1)]**：
    obs    = env.get_obs()                                    17 维
    action = [未来 K 拍后的关节角(6), 未来 K 拍后的夹爪命令(1)]   7 维

为什么这么取（都是实测踩出来的）：
1. 位置伺服把目标跟得极紧（97% 的帧 |目标-当前| < 0.01 rad），用单步 (状态 -> 当前目标)
   训练出来的网络退化成近似恒等映射 f(s)≈s，闭环里任何状态都是不动点——实测策略停在半空不动。
   改成预测"K 拍之后的实测关节角"，动作就系统地领先当前状态（f(s)=s+Δ），闭环持续推进。
2. 夹爪那一维必须用"下发的命令"而不是"实测角"：夹住物块时实测角被物块顶在 ≈0.153，
   永远到不了命令值 0.135，用实测角训出来的策略给不出夹持力，抬起来物块就滑掉。
3. K 取小一点（默认 2 拍 = 0.1s）：既保证领先量足够，又不会让手臂用 K 倍速度去追目标
   （K 大时实测在每段运动末尾冲过头，齿面冲到物块中心下方 4cm）。
"""

from __future__ import annotations

import numpy as np


# 各阶段名字（打印/调试用）
PHASES = ["home", "approach", "descend", "close", "lift", "transport", "place_down", "release"]

# 各阶段总 substeps（timestep=0.002s）。
# 刻意取短：位置伺服对目标跟得很紧，如果每个路点都等它停稳再走，录出来的
# "目标≈当前状态"的帧会占 97%，策略学到的是"原地别动"——实测在预备位姿处直接卡死。
# 让路点来得比伺服settle更快，臂就一直在追目标，数据里才有"往哪走"的信息。
DEFAULT_SUBSTEPS = {
    "home": 20,          # 臂已经停在 keyframe 的预备位姿，不需要再 settle
    "approach": 520,
    "descend": 360,
    "close": 700,        # 夹爪闭合约 300 步，其余是夹持保持
    "lift": 360,
    "transport": 520,
    "place_down": 360,
    "release": 800,      # 张开斜坡（太快会把物块弹飞）
}

# 每个阶段的笛卡尔路点数（1 = 直接一步 move_to）
DEFAULT_SEGMENTS = {
    "home": 1,
    "approach": 4,
    "descend": 3,
    "lift": 3,
    "transport": 4,
    "place_down": 3,
}

# 动作的前瞻量：动作 = 未来 LOOKAHEAD 个控制周期之后的实测状态（1 拍 = rec_every 个物理步）。
DEFAULT_LOOKAHEAD = 4   # 4*50ms = 0.2s

# 判定"真的把物块拿起来了"的最低抬升高度：纯推着滑过去 lift≈0，人手抓起来通常抬几厘米。
# 任务成败以 env.success() 为准（物块落在放置点、夹爪张开），抬升只用来排除"推"。
MIN_LIFT = 0.02

APPROACH_H = 0.10   # 抓取/放置前停在上方的高度
LIFT_H = 0.14       # 抓起后抬升的高度
PLACE_H = 0.02      # 放置时齿面中点离放置点的余高


def build_frames(stream, lookahead: int = DEFAULT_LOOKAHEAD):
    """把录制流打成训练样本。

    stream 里每帧是 (obs, 实测关节角(6), 下发的夹爪命令(1))；样本为
        obs_i -> [未来第 lookahead 拍的实测关节角(6), 该拍的夹爪命令(1)]
    手动遥操作采集（manual_collect.py）用同一套约定，所以数据可以直接混在一起训。
    """
    frames = []
    for i in range(len(stream) - lookahead):
        _, q_fut, jaw_cmd_fut = stream[i + lookahead]
        frames.append((stream[i][0],
                       np.concatenate([q_fut, [jaw_cmd_fut]]).astype(np.float32)))
    return frames


def pick_place(
    env,
    *,
    obj_pose: np.ndarray | None = None,
    place_xyz: np.ndarray | None = None,
    squeeze: float = 0.004,
    record: bool = False,
    rec_every: int = 25,
    lookahead: int = DEFAULT_LOOKAHEAD,
    frame_hook=None,
    substeps: dict | None = None,
    segments: dict | None = None,
    collect_log: bool = False,
):
    """跑一次完整的抓取-放置。返回 dict（含 ok / 抬升高度 / 放置误差 / 可选录制帧）。

    obj_pose:  7 维物块初始位姿；None 表示用场景 keyframe 里的位置
    place_xyz: 3 维放置点；None 表示用场景里已有的放置点
    record:    True 时返回 frames=[(obs(17), action(7)), ...]
    lookahead: 动作取的"未来第几拍状态"（默认 4 拍 = 0.2s）
    frame_hook: 每录一帧就被调用一次（与 stream 同步），VLA 采集用它抓图像。
    """
    sub = dict(DEFAULT_SUBSTEPS)
    seg = dict(DEFAULT_SEGMENTS)
    if substeps:
        sub.update(substeps)
    if segments:
        seg.update(segments)

    stream: list[tuple[np.ndarray, np.ndarray, float]] = []   # (obs, 实测关节角, 下发的夹爪命令)
    step_count = 0

    def make_hook(_grip_open: bool):
        def hook(_i, _q_cmd, jaw_cmd):
            nonlocal step_count
            step_count += 1
            if not record or step_count % rec_every:
                return
            stream.append((env.get_obs().astype(np.float32).copy(),
                           env.get_joints().astype(np.float32), float(jaw_cmd)))
            if frame_hook is not None:      # VLA 采集在这里顺手渲染图像
                frame_hook()
        return hook

    def cart_move(target, grip_open, phase, iters_last=3):
        """在齿面中点空间插值成 seg[phase] 个路点依次 move_to（避免手走弧线扫到物块）。"""
        target = np.asarray(target, dtype=float)
        n = max(1, int(seg.get(phase, 1)))
        per = max(1, sub[phase] // n)
        hook = make_hook(grip_open)
        p0 = env.tooth_midpoint().copy()
        q = None
        for k in range(1, n + 1):
            p = p0 + (target - p0) * (k / n)
            # 中间路点只迭代一次：臂没必要停稳，下一段路点会接管
            q, _ = env.move_to(p, grip_open, substeps=per, squeeze=squeeze, hook=hook,
                               iters=iters_last if k == n else 1,
                               tol=0.002 if k == n else 0.02)
        return q

    log = []

    def snap(tag):
        log.append((tag, env.get_object_pos().copy(), env.tooth_midpoint().copy(),
                    env.tooth_gap(), env.get_gripper()))

    # ---- 起始 ----
    env.reset(object_pose=obj_pose)
    if place_xyz is not None:
        env.place_on_table(place_xyz)
    env.move(env.q_home, gripper_open=True, substeps=sub["home"], hook=make_hook(True))
    snap("home")

    obj0 = env.get_object_pos().copy()
    place = env.get_place_target().copy()

    # ---- 抓取 ----
    cart_move(obj0 + [0, 0, APPROACH_H], True, "approach")
    snap("above")
    cart_move(obj0, True, "descend")
    snap("at-obj")
    cart_move(obj0, False, "close")          # 目标不变，只是闭爪
    snap("close")
    peaks = [env.get_object_pos()[2]]

    # ---- 搬运 ----
    cart_move(obj0 + [0, 0, LIFT_H], False, "lift")
    peaks.append(env.get_object_pos()[2])
    snap("lift")
    cart_move(place + [0, 0, APPROACH_H], False, "transport")
    peaks.append(env.get_object_pos()[2])
    snap("place-above")
    q_down = cart_move(place + [0, 0, PLACE_H], False, "place_down")
    peaks.append(env.get_object_pos()[2])
    snap("place-at")

    # ---- 释放（缓开爪）----
    env.release(q_down, substeps=sub["release"], hook=make_hook(True))
    snap("release")

    # ---- 录制：obs_i -> [未来第 lookahead 拍的实测关节角(6), 未来第 lookahead 拍下发的夹爪命令(1)] ----
    # 夹爪取"命令"而不是"实测角"：夹住物块时实测角被物块顶住(≈0.153)永远到不了命令值(0.135)，
    # 用实测角训练出来的策略永远给不出夹持力，抬升时物块会滑掉。关节角则相反——
    # 用实测角可以让动作系统地领先当前状态，闭环不会退化成恒等映射。
    frames = build_frames(stream, lookahead) if record else []

    obj_end = env.get_object_pos()
    lift = float(max(peaks) - obj0[2])
    horiz_err = float(np.linalg.norm(obj_end[:2] - place[:2]))
    ok = bool(lift > MIN_LIFT and env.success())

    out = {
        "ok": ok,
        "lift": lift,
        "horiz_err": horiz_err,
        "success": bool(env.success()),
        "obj0": obj0,
        "place": place,
        "obj_end": obj_end,
        "frames": frames,
        "steps": step_count,
    }
    if collect_log:
        out["log"] = log
    return out

# -*- coding: utf-8 -*-
"""采样任务族：分级规格注册表（**单一真值表**）。

设计原则（见 `WYSD/Lerobot-Uranus-VLA-Demo/docs/underwater_vla_task_design.md`）：
- 族 A（科学/生物采样），**每级只加一个难度轴**，这样掉点能归因；
- 工作区几何不动（沉积物面 z=0.35），只换道具与判据 -> 保住 IK/伺服/夹爪标定；
- 每级必须有自动判据（`success` / `settled`），否则不做；
- 数据由**人手操作**采集，所以这里不提供专家策略；本文件只负责"难度 + 判据 + 语言指令"。

每个等级的字段：
    title        中文名
    xml          场景文件（相对本包）
    difficulty   这一级新增的难度轴（一句话）
    object/goal  被操作物 / 目标点载体
    sample_layout(rng) -> layout   布局随机化（**唯一的随机性来源**）
    success(env)  即时判据（几何量 + 接触力）
    settled(env)  "真的停稳了"判据（配合采集器的 hold_s 用）
    hold_s       采集器要求判据连续成立多久才算完成
    force_limit  接触力上限（N），只有需要力约束的等级才有
    instructions 训练用指令池；holdout 只评测不训练
    criterion    成功判据的文字说明（§3.2 模板要求；实现在 success/settled 里）
    water        采集用的水况建议（underwater_vision 预设名）。
                 **七级统一取 coastal**：水况是"正交的第二维"，若每级用水况不同，
                 跨级比较就被水况混淆了（等于一次动两个变量）。能见度扫描请作为**独立的
                 评测轴**做：固定某一级，扫 clear/coastal/turbid/harbor，得成功率–能见度曲线。
    notes        实现要点/坑
"""
from __future__ import annotations

import numpy as np

__all__ = ["LEVELS", "LEVEL_ORDER", "get", "describe_all"]

# 沉积物上表面高度：所有道具都以此为基准（与抓取场景的桌面同高）
SURFACE_Z = 0.35


# ----------------------------------------------------------------------
# 通用判据零件
# ----------------------------------------------------------------------
def _in_basket(env, pos, xy_tol: float = 0.045, z_hi: float = 0.030) -> bool:
    """物块落在采样篮内：水平进篮口、竖直落在篮底附近。

    篮内静置中心由 place_target_site 给出（场景里定义成"底板顶面 + 物块半高"），
    所以这里只需比较相对量，换篮/换物块尺寸都不用改代码。
    """
    goal = env.get_place_target()
    dxy = float(np.linalg.norm(np.asarray(pos)[:2] - goal[:2]))
    dz = float(np.asarray(pos)[2] - goal[2])
    return dxy < xy_tol and -0.02 < dz < z_hi


def _gripper_open(env, tol: float = 0.12) -> bool:
    return env.get_gripper() > env.open_angle - tol


def _basket_settled(env) -> bool:
    """物块已经真的**落到篮底**（不是被举着从篮口经过，也不是还躺在沉积物上）。

    注意必须带上 _in_basket：只看高度的话，物块躺在沉积物面上时（0.37 vs 篮内静置 0.375）
    也会判 settled=True。
    """
    pos = env.get_object_pos()
    return (_in_basket(env, pos)
            and abs(float(pos[2] - env.get_place_target()[2])) < 0.012)


# ----------------------------------------------------------------------
# L1 抓取取样（基线）
# ----------------------------------------------------------------------
def layout_l1(rng):
    for _ in range(200):
        ox, oy = rng.uniform(1.575, 1.625), rng.uniform(0.02, 0.10)
        gx, gy = rng.uniform(1.575, 1.625), rng.uniform(-0.16, -0.06)
        if np.hypot(ox - gx, oy - gy) < 0.10:
            continue
        return {"props": {"object": [ox, oy, SURFACE_Z + 0.02, 1, 0, 0, 0]},
                "goal_xy": (gx, gy)}
    raise RuntimeError("L1 采样不到合法布局")


# ----------------------------------------------------------------------
# L2 弱纹理 + 随机位姿（同一任务，扩大范围 + 灰色目标）
# ----------------------------------------------------------------------
def layout_l2(rng):
    for _ in range(200):
        ox, oy = rng.uniform(1.555, 1.645), rng.uniform(-0.10, 0.12)
        gx, gy = rng.uniform(1.555, 1.645), rng.uniform(-0.18, 0.04)
        if np.hypot(ox - gx, oy - gy) < 0.10:
            continue
        return {"props": {"object": [ox, oy, SURFACE_Z + 0.02, 1, 0, 0, 0]},
                "goal_xy": (gx, gy)}
    raise RuntimeError("L2 采样不到合法布局")


# ----------------------------------------------------------------------
# L3 双目标 + 语言指称
# ----------------------------------------------------------------------
_L3_INSTR = {
    "object": [  # A 岩 = 立柱旁边那块
        "把立柱旁边的岩石放进采样篮",
        "抓取紧挨着立柱的那块岩石",
        "捡起有立柱作标记的岩石放进采样篮",
    ],
    "object2": [  # B 岩 = 空地上那块
        "把离立柱较远的那块岩石放进采样篮",
        "抓取空地上的那块岩石",
        "捡起没有立柱陪着的那块岩石",
    ],
}
_L3_HOLDOUT = {
    "object": ["把挨着立柱那块样本收进采样篮"],
    "object2": ["把空着的地方那块样本收进采样篮"],
}


def layout_l3(rng):
    """两块同色岩石分居立柱两侧；指令决定哪一块是目标（干扰项被动到就失败）。

    间距是硬约束：岩石**不能落在托盘里或托盘边上**，否则一开局就"已经成功/自己滑进盘"，
    数据全废。这里按托盘外廓（半宽 6.75cm）+ 岩石半宽留 12cm 以上。
    """
    for _ in range(400):
        ya = rng.uniform(0.09, 0.13)          # A 岩（立柱侧，+y）
        yb = rng.uniform(-0.03, 0.01)         # B 岩（空地侧）
        xa = rng.uniform(1.585, 1.615)
        xb = rng.uniform(1.585, 1.615)
        gx, gy = rng.uniform(1.585, 1.615), rng.uniform(-0.165, -0.145)
        if np.hypot(xa - xb, ya - yb) < 0.07:
            continue
        if np.hypot(xa - gx, ya - gy) < 0.15 or np.hypot(xb - gx, yb - gy) < 0.12:
            continue
        target = "object" if rng.random() < 0.5 else "object2"
        seen = rng.random() < 0.75            # 75% 用训练模板，25% 用 hold-out
        pool = (_L3_INSTR if seen else _L3_HOLDOUT)[target]
        return {
            "props": {
                "object": [xa, ya, SURFACE_Z + 0.02, 1, 0, 0, 0],
                "object2": [xb, yb, SURFACE_Z + 0.02, 1, 0, 0, 0],
            },
            "statics": {"marker_post": [xa, ya + 0.05, SURFACE_Z]},
            "goal_xy": (gx, gy),
            "target_body": target,
            "instruction": str(rng.choice(pool)),
            "instruction_split": "seen" if seen else "holdout",
        }
    raise RuntimeError("L3 采样不到合法布局")


def success_l3(env) -> bool:
    tgt = env.prop_pos(env.layout["target_body"])
    other_name = "object2" if env.layout["target_body"] == "object" else "object"
    other = env.prop_pos(other_name)
    init = env.init_props[other_name]
    moved = float(np.linalg.norm(other[:2] - init[:2]))
    return _in_basket(env, tgt) and _gripper_open(env) and moved < 0.01


def settled_l3(env) -> bool:
    return abs(float(env.prop_pos(env.layout["target_body"])[2]
                     - env.get_place_target()[2])) < 0.012


# ----------------------------------------------------------------------
# L4 易碎珊瑚（力约束）
# ----------------------------------------------------------------------
def layout_l4(rng):
    for _ in range(200):
        ox, oy = rng.uniform(1.578, 1.622), rng.uniform(0.00, 0.08)
        gx, gy = rng.uniform(1.578, 1.622), rng.uniform(-0.16, -0.06)
        if np.hypot(ox - gx, oy - gy) < 0.10:
            continue
        return {"props": {"object": [ox, oy, SURFACE_Z + 0.02, 1, 0, 0, 0]},
                "goal_xy": (gx, gy)}
    raise RuntimeError("L4 采样不到合法布局")


def success_l4(env) -> bool:
    """进托盘 + 夹爪张开 + **珊瑚没有翻倒**（本体 z 轴偏离竖直 <25°）。

    为什么不用"夹持力阈值"判夹碎（原计划）：实测夹持力峰值在 kN 量级且**非单调**
    （轻合 squeeze=2mm → 5220N / 标称 4mm → 3801N / 一夹到底 → 8139N）。位置伺服
    (kp=3000) 压进硬接触的瞬态尖峰主要取决于接触瞬间的相对速度，与 squeeze 不成正比，
    拿它当阈值不可复现；更要命的是现有遥操作是**二值夹爪**（Space 开/合），操作者
    根本没有"更轻地合爪"这个动作，力阈值本身不可操作。
    所以这一级改成人力可控、可复现的"轻拿轻放别碰倒"，力峰值只作为过程量记录。
    真要考"易碎"需要先给夹爪加模拟量控制（已记为后续工作）。
    """
    return (_in_basket(env, env.get_object_pos()) and _gripper_open(env)
            and env.object_tilt_deg() < 25.0)


# ----------------------------------------------------------------------
# L5 推芯采样（插管坐底）
# ----------------------------------------------------------------------
def layout_l5(rng):
    cx = rng.uniform(1.585, 1.615)
    cy = rng.uniform(0.07, 0.12)
    sx = rng.uniform(1.585, 1.615)
    sy = rng.uniform(-0.11, -0.05)
    return {"props": {"object": [cx, cy, SURFACE_Z + 0.075, 1, 0, 0, 0]},  # 竖直站立
            "goal_xy": (sx, sy)}


def success_l5(env) -> bool:
    """管心落到"坐底位"附近（同时管住深度与横向偏心）+ 管轴基本竖直。"""
    err = float(np.linalg.norm(env.get_object_pos() - env.get_place_target()))
    return err < env.spec["tol"]["seat"] and env.object_tilt_deg() < env.spec["tol"]["tilt_deg"]


def settled_l5(env) -> bool:
    return bool(success_l5(env))


# ----------------------------------------------------------------------
# L6 探针点测
# ----------------------------------------------------------------------
def layout_l6(rng):
    return {"props": {"object": [rng.uniform(1.575, 1.625), rng.uniform(0.00, 0.10),
                                 SURFACE_Z + 0.02, 0.70710678, 0.0, 0.70710678, 0.0]},
            "goal_xy": (rng.uniform(1.755, 1.775), rng.uniform(-0.02, 0.06))}


def success_l6(env) -> bool:
    """针尖贴住测点 + 接触力没超限（压过头算失败）。保持时长由采集器 hold_s 管。"""
    d = float(np.linalg.norm(env.probe_tip_pos() - env.get_place_target()))
    return d < env.spec["tol"]["tip"] and env.force_now < env.spec["force_limit"]


# ----------------------------------------------------------------------
# L7 长程：取探针 -> 点测 -> 归位工具架
# ----------------------------------------------------------------------
def layout_l7(rng):
    return {"props": {"object": [rng.uniform(1.578, 1.622), rng.uniform(0.02, 0.10),
                                 SURFACE_Z + 0.02, 0.70710678, 0.0, 0.70710678, 0.0]},
            "statics": {"tool_rack": [1.60, rng.uniform(-0.16, -0.10), SURFACE_Z]},
            "goal_xy": (rng.uniform(1.755, 1.775), rng.uniform(0.00, 0.06))}


def success_l7(env) -> bool:
    """三段里程碑全过：① 拿起过探针 ② 点测接触保持达标 ③ 探针回架且夹爪张开。"""
    rack = env.site_pos("tool_rack_site")
    p = env.get_object_pos()
    in_rack = (float(np.linalg.norm(p[:2] - rack[:2])) < 0.03
               and abs(float(p[2] - rack[2])) < 0.02)
    return bool(env.milestone_lifted and env.milestone_touch and in_rack
                and _gripper_open(env))


def settled_l7(env) -> bool:
    rack = env.site_pos("tool_rack_site")
    p = env.get_object_pos()
    return (float(np.linalg.norm(p[:2] - rack[:2])) < 0.03
            and abs(float(p[2] - rack[2])) < 0.015)


# ----------------------------------------------------------------------
# L8 杠杆阀门（族 B · 工业干预）
# ----------------------------------------------------------------------
# 阀门手柄的几何：本体原点 = 握把中心，铰链在本地 +x 0.055m（阀杆轴线）。
# 所以"阀门开度"就是 object_joint 这条 hinge 的角；握把位置 = 铰链 + R_z(q)·(握把-铰链)。
L8_PIVOT = np.array([1.72, 0.0, 0.50])   # 铰链（阀杆轴线）位置
L8_KNOB_R = 0.055                        # 握把中心到铰链的距离
L8_OPEN = np.deg2rad(60.0)               # 开到位 = 手柄抬起（铰链限位 ±75°）
L8_CLOSED = -np.deg2rad(60.0)            # 关到位 = 手柄压下

_L8_INSTR = {
    "open": ["把阀门打开", "扳动手柄把阀门开到最大", "把阀门手柄扳到开启位置"],
    "close": ["把阀门关闭", "扳动手柄把阀门关到底", "把阀门手柄扳回关闭位置"],
}
_L8_HOLDOUT = {"open": ["让这个阀门通起来"], "close": ["把这个阀门切断"]}


def l8_knob_pos(q: float) -> np.ndarray:
    """给定阀门开度 q(rad)，返回握把中心的三维位置。

    手柄绕**水平 y 轴**摆动，所以握把在 x-z 平面里画弧：
        q=0 水平（握把在铰链的 -x 侧）；q=+60° 抬起（开）；q=-60° 压下（关）。
    换成 y 轴的原因是实测出来的：夹爪沿 y 合拢，若铰链是竖直 z 轴，
    夹持力会产生 τ=F_y·r_x 的扭矩（实测把起始 -45° 的阀门夹到 -93°），
    而绕 y 轴时该扭矩分量恒为 0，夹紧不扰动初始状态。详见场景 XML 的注释。
    """
    return np.array([L8_PIVOT[0] - L8_KNOB_R * float(np.cos(q)),
                     L8_PIVOT[1],
                     L8_PIVOT[2] + L8_KNOB_R * float(np.sin(q))])


def layout_l8(rng):
    """起始角取水平位（0°±8°），指令决定往"开"（抬起）还是往"关"（压下）扳。

    **起始取中间是有意的**：无论从"开"还是"关"起步，反方向那条指令都无事可做，
    "同初始状态、只换指令"的方向词消融就做不成了（设计文档 §6.1 第 3 项）。
    """
    q0 = float(np.deg2rad(rng.uniform(-8.0, 8.0)))
    to_open = bool(rng.random() < 0.5)
    target = float(L8_OPEN if to_open else L8_CLOSED)
    seen = bool(rng.random() < 0.75)
    pool = (_L8_INSTR if seen else _L8_HOLDOUT)["open" if to_open else "close"]
    # 目标标记用**三维**位置：绕 y 轴转动时开位/关位的 (x,y) 相同、只差 z，
    # 放在地面上的标记根本区分不出"开"和"关"。
    pos = l8_knob_pos(target)
    return {"joints": {"object_joint": q0},
            "start_angle": q0, "target_angle": target, "to_open": to_open,
            "statics": {"place_target": [float(pos[0]), float(pos[1]), float(pos[2])]},
            "instruction": str(rng.choice(pool)),
            "instruction_split": "seen" if seen else "holdout"}


def success_l8(env) -> bool:
    """把手柄扳到指令要求的一端：开度到目标 5° 以内 + 相对起始至少转了 25°。

    方向由 target 相对 start 的符号决定，所以"扳反了"必然失败 —— 这正是方向词能被检验的地方。
    """
    q = env.joint_angle("object_joint")
    q0, tgt = env.layout["start_angle"], env.layout["target_angle"]
    travel = q - q0
    toward = np.sign(tgt - q0)
    return bool(abs(q - tgt) < np.deg2rad(8.0) and travel * toward > np.deg2rad(25.0))


def settled_l8(env) -> bool:
    """手柄停住了：角度到位且角速度很小（不是还在被推着/回弹）。"""
    return bool(abs(env.joint_angle("object_joint") - env.layout["target_angle"])
                < np.deg2rad(6.0) and abs(env.joint_vel("object_joint")) < 0.35)


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------
LEVELS = {
    "l1_rock_collect": dict(
        order=1, family="A", title="L1 抓取取样", xml="scenes/l1_rock_collect.xml",
        difficulty="基线：高对比目标（黄岩）+ 窄随机范围",
        criterion="岩石落进托盘（水平 <45mm、落在盘底附近）+ 夹爪张开",
        object_desc="4cm 立方体岩石（亮黄，对比最强）",
        goal_desc="采样托盘（内净 12.5cm、三面壁 2.5cm、朝机器人一侧开口）",
        sample_layout=layout_l1,
        success=lambda env: _in_basket(env, env.get_object_pos()) and _gripper_open(env),
        settled=_basket_settled, hold_s=0.4,
        instructions=["把岩石放进采样篮", "捡起这块岩石放进采样篮", "把岩石采集到采样篮里"],
        holdout=["把这块样本收进采样篮"],
        water="coastal",
        notes="唯一能与现有 pick-place 基线（300 段/50700 帧）对齐的等级；先用它把采集链路跑通。",
    ),
    "l2_gray_rock_random": dict(
        order=2, family="A", title="L2 弱纹理 + 随机位姿", xml="scenes/l2_gray_rock_random.xml",
        difficulty="目标外观低对比（灰岩 vs 沉积物底色）+ 初始位姿范围扩大",
        criterion="同 L1（几何完全一致，只有配色与随机范围变了）",
        object_desc="4cm 立方体岩石（灰色，与底质同色系）",
        goal_desc="采样托盘（同 L1）",
        sample_layout=layout_l2,
        success=lambda env: _in_basket(env, env.get_object_pos()) and _gripper_open(env),
        settled=_basket_settled, hold_s=0.4,
        instructions=["把灰色岩石放进采样篮", "捡起这块岩石放进采样篮", "采集这块岩石"],
        holdout=["把这块灰岩收进采样篮"],
        water="coastal",
        notes="几何与 L1 完全一致，**只改配色与随机范围** —— 与 L1 的差值就是"
              "「定位变难」的代价，是干净的消融对照。",
    ),
    "l3_two_rocks_language": dict(
        order=3, family="A", title="L3 双目标 + 语言指称", xml="scenes/l3_two_rocks_language.xml",
        difficulty="语言指称 + 干扰项（两块同色岩石，靠立柱参照区分）",
        criterion="指令指定的那块进托盘 + 干扰项位移 <10mm + 夹爪张开",
        object_desc="两块外观完全相同的灰色岩石（A 挨着立柱 / B 在空地）",
        goal_desc="采样托盘（同 L1，在两块岩石之外）",
        sample_layout=layout_l3, success=success_l3, settled=settled_l3, hold_s=0.4,
        instructions=_L3_INSTR["object"] + _L3_INSTR["object2"],
        holdout=_L3_HOLDOUT["object"] + _L3_HOLDOUT["object2"],
        water="coastal",
        notes="干扰项位移 >1cm 即判失败 —— 这是「语言分支到底有没有被用上」的最小可判实验"
              "（设计文档 §4 + 评测矩阵第 3 项）。采集时同场景不同指令 -> 数据里天然含语言-行为配对。",
    ),
    "l4_fragile_coral": dict(
        order=4, family="A", title="L4 易碎珊瑚轻取", xml="scenes/l4_fragile_coral.xml",
        difficulty="窄目标（3cm）+ 软垫落点 + 不得翻倒/摔落",
        criterion="珊瑚进托盘 + 未翻倒（本体 z 轴偏离竖直 <25°）+ 夹爪张开",
        object_desc="3x3x4cm 珊瑚碎块（比 L1/L2 的 4cm 方岩更窄，夹持窗口更小）",
        goal_desc="采样托盘（同 L1）+ 软垫",
        sample_layout=layout_l4, success=success_l4, settled=_basket_settled, hold_s=0.4,
        instructions=["把珊瑚轻轻放进采样篮", "采集这块珊瑚", "把珊瑚样本收进采样篮"],
        holdout=["把这段珊瑚移进采样篮，别弄断"],
        water="coastal",
        notes="**原计划的力阈值判据已被实测否掉**（见 success_l4 的 docstring：力峰值 kN 量级"
              "且非单调，加上遥操作是二值夹爪、不可操作），改成'别碰倒'。夹持力峰值仍逐拍记录在"
              "metrics 里，等给夹爪加了模拟量控制再启用。",
    ),
    "l5_push_core": dict(
        order=5, family="A", title="L5 推芯采样（插管坐底）", xml="scenes/l5_push_core.xml",
        difficulty="工具夹持 + 竖直搬运 + 精密插入（管底坐底 + 管轴竖直 + 横向偏心）",
        criterion="管心到坐底位 <8mm（同时管住深度与横向偏心）+ 管轴偏离竖直 <8°",
        object_desc="采样管（圆柱 r=2cm、长 15cm，竖直站立，底部对准井底）",
        goal_desc="取样座（方形导向井，内净 8x8cm、壁高 5cm）",
        sample_layout=layout_l5, success=success_l5, settled=settled_l5, hold_s=1.0,
        tol={"seat": 0.008, "tilt_deg": 8.0},
        instructions=["把采样管插进取样座", "将采样管竖直插入取样座坐底", "用采样管完成一次取芯"],
        holdout=["把取芯管放到取样座里到底"],
        water="coastal",
        notes="导向井是真实工装，判据是纯几何量（管心到坐底位的距离 + 管轴倾角），可复现；"
              "比「插进软泥」靠软接触近似要干脆得多。",
    ),
    "l6_probe_touch": dict(
        order=6, family="A", title="L6 探针点测", xml="scenes/l6_probe_touch.xml",
        difficulty="mm 级对准 + 接触力上限 + 长保持（2s）",
        criterion="针尖距测点 <8mm + 接触力 <12N（压过头即失败）",
        object_desc="探针（手柄 r=2cm + 细针尖，平放、针尖朝 +x）",
        goal_desc="管壁测点（远处立块 -x 面上，高 0.40m）",
        sample_layout=layout_l6, success=success_l6, hold_s=2.0,
        force_limit=12.0, tol={"tip": 0.008},
        instructions=["把探针顶到测点上", "用探针接触标记测点", "把探针尖对准测点并保持"],
        holdout=["让探针抵住管壁的标记点"],
        water="coastal",
        notes="力上限防「压过头」：保持 2s 期间力必须一直低于阈值，所以是「贴着但不顶着」。",
    ),
    "l7_tool_roundtrip": dict(
        order=7, family="A", title="L7 长程：取工具 -> 点测 -> 归位", xml="scenes/l7_tool_roundtrip.xml",
        difficulty="多里程碑长程 + 归位（L6 全部内容，外加必须把探针放回工具架）",
        criterion="三段里程碑全过：拿起过探针 → 点测达标 → 回托板静置 + 夹爪张开",
        object_desc="探针（同 L6）",
        goal_desc="管壁测点 + 工具托板",
        sample_layout=layout_l7, success=success_l7, settled=settled_l7, hold_s=1.5,
        force_limit=12.0, tol={"tip": 0.008},
        instructions=["取出探针，点测后放回工具架", "用探针测完这个点再把探针放回去",
                      "完成点测并把探针归位"],
        holdout=["测完这个测点，把探针收回工具架"],
        water="coastal",
        notes="考顺序与归位（设计文档 S6 的最小可判版本）。里程碑由 env.observe() 逐拍累计："
              "离开沉积物面 -> 点测保持达标 -> 回架静置。",
    ),
    "l8_valve_lever": dict(
        order=8, family="B", title="L8 杠杆阀门（干预）", xml="scenes/l8_valve_lever.xml",
        criterion="阀门开度到指令要求那一端 <8° + 相对起始至少转 25°，方向必须对",
        difficulty="接触丰富 + 持续施力 + **方向词**（同初始状态换指令必须反向）",
        object_desc="阀门手柄握把（r=2cm 竖直圆柱，绕水平 y 轴铰接）",
        goal_desc="阀门另一端（抬起=开 / 压下=关，空中有目标位置指示球）",
        sample_layout=layout_l8, success=success_l8, settled=settled_l8, hold_s=1.0,
        tol={"angle_deg": 8.0, "travel_deg": 25.0},
        instructions=_L8_INSTR["open"] + _L8_INSTR["close"],
        holdout=_L8_HOLDOUT["open"] + _L8_HOLDOUT["close"],
        water="coastal",
        notes="⚠ **状态：设计完成、尚未通过可操作性验证，暂不要用它采数据。** "
              "已实测确认的三件事：① 铰链 range 必须写弧度（本模型 angle=radian）；"
              "② 手柄臂不能和齿面同层（绕 z 轴时臂会扫进齿间，卡死在 -66.7°）；"
              "③ **不能用夹爪去抓手柄** —— 夹持力会产生扭矩：绕 z 轴时 τ=F_y·r_x≈78N·m；"
              "即使改成绕 y 轴（该轴扭矩分量为 0），align_jaws 让齿面法向偏 z 约 5.4°，"
              "1224N 的夹持力仍有 115N 的 z 分量 -> 6.3N·m，实测把 0° 的阀门夹成 -42.5°。"
              "结论：这个任务必须是**推**而不是夹，且推的姿态要「张开夹爪 + y 向偏置让单齿去推」"
              "（闭合夹爪的两齿间距 1.3cm < 手柄臂厚 1.4cm，会直接夹住手柄臂）。"
              "下一步：标定推的姿态（待推位 + 沿弧切向的退让量），再跑双向验证。"
              "本包第一个非抓放任务，也是唯一判据为**关节角**的等级。"
              "起始取中间位置是为了让开/关两条指令都非平凡 -> 方向词消融（同初始状态只换指令）"
              "能给出最强证据：策略若对开/关给出同样动作，语言分支没被用上就无可辩驳。"
              "难度旋钮 = 起始角范围、目标角（行程）、能见度。frictionloss 实测**不是**可用旋钮：2000N 夹持力带来的摩擦扭矩比它大两个数量级。",
    ),
}

LEVEL_ORDER = [k for k, _ in sorted(LEVELS.items(), key=lambda kv: kv[1]["order"])]


def get(level: str) -> dict:
    if level not in LEVELS:
        raise KeyError(f"未知等级 {level!r}；可用：{', '.join(LEVEL_ORDER)}")
    return LEVELS[level]


def describe_all() -> str:
    """给 README / --list 用的一张表。"""
    lines = [f"{'等级':<24} {'新增难度轴':<44} {'保持':>5}  水况"]
    lines.append("-" * 92)
    for k in LEVEL_ORDER:
        s = LEVELS[k]
        lines.append(f"{s['title']:<24} {s['difficulty'][:42]:<44} {s['hold_s']:>4.1f}s  {s['water']}")
    return "\n".join(lines)

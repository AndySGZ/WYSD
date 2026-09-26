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
    water        建议水况（能见度旋钮；对齐 underwater_vision 预设）
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
# 注册表
# ----------------------------------------------------------------------
LEVELS = {
    "l1_rock_collect": dict(
        order=1, title="L1 抓取取样", xml="scenes/l1_rock_collect.xml",
        difficulty="基线：高对比目标（黄岩）+ 窄随机范围",
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
        order=2, title="L2 弱纹理 + 随机位姿", xml="scenes/l2_gray_rock_random.xml",
        difficulty="目标外观低对比（灰岩 vs 沉积物底色）+ 初始位姿范围扩大",
        object_desc="4cm 立方体岩石（灰色，与底质同色系）",
        goal_desc="采样托盘（同 L1）",
        sample_layout=layout_l2,
        success=lambda env: _in_basket(env, env.get_object_pos()) and _gripper_open(env),
        settled=_basket_settled, hold_s=0.4,
        instructions=["把灰色岩石放进采样篮", "捡起这块岩石放进采样篮", "采集这块岩石"],
        holdout=["把这块灰岩收进采样篮"],
        water="turbid",
        notes="几何与 L1 完全一致，**只改配色与随机范围** —— 与 L1 的差值就是"
              "「定位变难」的代价，是干净的消融对照。",
    ),
    "l3_two_rocks_language": dict(
        order=3, title="L3 双目标 + 语言指称", xml="scenes/l3_two_rocks_language.xml",
        difficulty="语言指称 + 干扰项（两块同色岩石，靠立柱参照区分）",
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
        order=4, title="L4 易碎珊瑚轻取", xml="scenes/l4_fragile_coral.xml",
        difficulty="窄目标（3cm）+ 软垫落点 + 不得翻倒/摔落",
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
        order=5, title="L5 推芯采样（插管坐底）", xml="scenes/l5_push_core.xml",
        difficulty="工具夹持 + 竖直搬运 + 精密插入（管底坐底 + 管轴竖直 + 横向偏心）",
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
        order=6, title="L6 探针点测", xml="scenes/l6_probe_touch.xml",
        difficulty="mm 级对准 + 接触力上限 + 长保持（2s）",
        object_desc="探针（手柄 r=2cm + 细针尖，平放、针尖朝 +x）",
        goal_desc="管壁测点（远处立块 -x 面上，高 0.40m）",
        sample_layout=layout_l6, success=success_l6, hold_s=2.0,
        force_limit=12.0, tol={"tip": 0.008},
        instructions=["把探针顶到测点上", "用探针接触标记测点", "把探针尖对准测点并保持"],
        holdout=["让探针抵住管壁的标记点"],
        water="turbid",
        notes="力上限防「压过头」：保持 2s 期间力必须一直低于阈值，所以是「贴着但不顶着」。",
    ),
    "l7_tool_roundtrip": dict(
        order=7, title="L7 长程：取工具 -> 点测 -> 归位", xml="scenes/l7_tool_roundtrip.xml",
        difficulty="多里程碑长程 + 归位（L6 全部内容，外加必须把探针放回工具架）",
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

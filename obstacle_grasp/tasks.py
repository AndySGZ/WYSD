# -*- coding: utf-8 -*-
"""障碍抓取族：分级规格注册表（**单一真值表**）。

与 `underwater_sampling`（水下族）**正交**：那一族考"看不清"（水体衰减/背向散射/低对比），
这一族考"看不见 / 够不着 / 路被堵"（遮挡、阻挡、窄缝、覆盖），**不做水下渲染**。
两族可以交叉组合（同一障碍场景换水况），得到"障碍 × 能见度"的二维矩阵。

设计原则与水下族一致：
- 每级只加**一个**难度轴，O1 是无障碍基线 -> O1 与其它级的成功率差就是"障碍的代价"；
- 工作区几何不动（桌面 z=0.35、近边 x=1.52），只加障碍物 —— 保住 IK/伺服/夹爪标定；
- 每级必须有自动判据。

字段：
    title/xml/family(="O")    名称与场景
    obstacle     这一级的障碍物（人话描述）
    difficulty   新增的难度轴（一句话）
    criterion    成功判据文字（实现在 success/settled）
    sample_layout(rng) -> {"props":{...}, "statics":{...}, "goal_xy":(x,y)}
                 **唯一的随机性来源**；障碍物随目标一起移动，保证难度关系恒定
    success(env) / settled(env)
    hold_s       判据需连续成立多久才算完成
    instructions/holdout  语言指令池（≥3 训练 + 1 hold-out）
    notes        实现要点/坑
"""
from __future__ import annotations

import numpy as np

__all__ = ["LEVELS", "LEVEL_ORDER", "TABLE_Z", "get", "describe_all"]

TABLE_Z = 0.35          # 台面上表面
OBJ_HALF = 0.02         # 目标方块半边长（4cm）
OBJ_Z = TABLE_Z + OBJ_HALF   # 静置中心 0.37

# 目标与圆盘的随机范围（沿用现有抓取场景实测可用的范围）
OBJ_X = (1.555, 1.645)
OBJ_Y = (-0.10, 0.12)
GOAL_X = (1.555, 1.645)
GOAL_Y = (-0.18, 0.04)
MIN_SEP = 0.10          # 目标与圆盘的最小平面距离


def _sample_xy(rng, obj_x=OBJ_X, obj_y=OBJ_Y):
    """抽一对合法的 (目标, 圆盘) 平面位置，保证隔开 MIN_SEP。"""
    for _ in range(300):
        ox, oy = rng.uniform(*obj_x), rng.uniform(*obj_y)
        gx, gy = rng.uniform(*GOAL_X), rng.uniform(*GOAL_Y)
        if np.hypot(ox - gx, oy - gy) >= MIN_SEP:
            return (float(ox), float(oy)), (float(gx), float(gy))
    raise RuntimeError("采样不到合法的目标/圆盘组合")


def _obj_pose(x, y, z=OBJ_Z):
    return [float(x), float(y), float(z), 1.0, 0.0, 0.0, 0.0]


# ----------------------------------------------------------------------
# O1 开阔基线
# ----------------------------------------------------------------------
def layout_o1(rng):
    (ox, oy), goal = _sample_xy(rng)
    return {"props": {"object": _obj_pose(ox, oy)}, "goal_xy": goal}


# ----------------------------------------------------------------------
# O2 视觉遮挡（立板挡在相机与目标之间）+ 动态干扰块
# ----------------------------------------------------------------------
def layout_o2(rng):
    (ox, oy), goal = _sample_xy(rng)
    return {"props": {"object": _obj_pose(ox, oy),
                      "distractor": _obj_pose(ox, oy + 0.08, TABLE_Z + 0.015)},
            # 立板固定在目标的相机侧 9cm 处（相机在 -y），遮挡关系随目标一起走
            # 立板底边贴台面 -> 中心 z = 0.35 + 半高(0.036)
            "statics": {"occluder": [ox, oy - 0.09, 0.386]},
            "goal_xy": goal}


def success_o2(env) -> bool:
    """搬到圆盘 + **干扰块没被撞移位**（>1cm 即失败）。"""
    d = float(np.linalg.norm(env.prop_pos("distractor")[:2] - env.init_props["distractor"][:2]))
    return bool(env.base_success() and d < 0.01)


# ----------------------------------------------------------------------
# O3 前方低梁（挡腕部通道）—— 必须重新定向腕部
# ----------------------------------------------------------------------
def layout_o3(rng):
    # x 范围收窄：低梁要完整落在台面上（x 近边 1.52）
    (ox, oy), goal = _sample_xy(rng, obj_x=(1.600, 1.645))
    return {"props": {"object": _obj_pose(ox, oy)},
            "statics": {"overhang": [ox - 0.055, oy, 0.420]},
            "goal_xy": goal}


# ----------------------------------------------------------------------
# O4 窄缝插取（缝宽 6.5cm，齿面外廓 5.5cm）
# ----------------------------------------------------------------------
def layout_o4(rng):
    (ox, oy), goal = _sample_xy(rng)
    return {"props": {"object": _obj_pose(ox, oy)},
            "statics": {"slot_wall": [ox, oy, 0.38]},
            "goal_xy": goal}


# ----------------------------------------------------------------------
# O5 覆盖堆叠（先搬开上面那块）
# ----------------------------------------------------------------------
def layout_o5(rng):
    (ox, oy), goal = _sample_xy(rng)
    return {"props": {"object": _obj_pose(ox, oy),
                      "cover": _obj_pose(ox, oy, OBJ_Z + 2 * OBJ_HALF + 0.0005)},
            "goal_xy": goal}


# ----------------------------------------------------------------------
# 公共判据零件
# ----------------------------------------------------------------------
def _on_target(env, tol=0.012) -> bool:
    """目标已经真的**落在台面圆盘上**（不是被举着从圆盘上方经过）。"""
    return bool(env.base_success() and abs(float(env.get_object_pos()[2]) - OBJ_Z) < tol)


# ----------------------------------------------------------------------
# 注册表
# ----------------------------------------------------------------------
_INSTR = {
    "o1_open_baseline": [
        "把方块放到蓝色圆盘上",
        "把黄色方块捡起来放到圆盘上",
        "把方块移到底座上那个蓝色圆盘上",
    ],
    "o2_visual_occlusion": [
        "把被挡板挡住的方块放到圆盘上",
        "绕过挡板把方块捡到圆盘上",
        "把挡板后面的方块放到圆盘上",
    ],
    "o3_front_overhang": [
        "绕过横梁把方块放到圆盘上",
        "从横梁下面把方块取出来放到圆盘上",
        "躲开横梁把方块抓到圆盘上",
    ],
    "o4_narrow_slot": [
        "把窄缝里的方块取出来放到圆盘上",
        "从两堵墙中间夹出方块放到圆盘上",
        "把缝里的方块夹到圆盘上",
    ],
    "o5_covered_stack": [
        "先把上面那块拿开，再把下面的方块放到圆盘上",
        "搬走覆盖的方块，然后把底下的方块放到圆盘上",
        "挪开压在上面的方块再抓下面的方块放到圆盘上",
    ],
}
_HOLDOUT = {
    "o1_open_baseline": ["把这块东西搁到圆盘上"],
    "o2_visual_occlusion": ["把藏在挡板后面那块挪到圆盘上"],
    "o3_front_overhang": ["从梁底下把方块掏出来放圆盘上"],
    "o4_narrow_slot": ["把卡在两墙之间的方块弄到圆盘上"],
    "o5_covered_stack": ["先清掉上面的，再收拾下面那块到圆盘上"],
}

LEVELS = {
    "o1_open_baseline": dict(
        order=1, family="O", title="O1 开阔基线", xml="scenes/o1_open_baseline.xml",
        obstacle="无障碍（对照组）",
        difficulty="基线：把 4cm 方块放到圆盘上，视野与路径都通畅",
        criterion="方块落到圆盘上（水平 <50mm）+ 夹爪张开 + 已落在台面",
        sample_layout=layout_o1, success=lambda env: _on_target(env),
        settled=lambda env: _on_target(env), hold_s=0.4,
        instructions=_INSTR["o1_open_baseline"], holdout=_HOLDOUT["o1_open_baseline"],
        notes="**这一级是标尺**：O2~O5 与它的成功率差就是「加一个障碍要付多少代价」。"
              "也是唯一能与现有 pick-place 基线（300 段 / 50700 帧）直接对齐的等级。",
    ),
    "o2_visual_occlusion": dict(
        order=2, family="O", title="O2 视觉遮挡", xml="scenes/o2_visual_occlusion.xml",
        obstacle="相机与目标之间的立板（顶 0.422m -> 视线掠顶落在目标高度正中，约 50% 被挡）"
                 " + 一个会动的干扰块",
        difficulty="**看得见但看不清**：目标被切掉一部分，路径仍通畅",
        criterion="方块落到圆盘上 + 干扰块位移 <10mm",
        sample_layout=layout_o2, success=success_o2, settled=lambda env: _on_target(env),
        hold_s=0.4,
        instructions=_INSTR["o2_visual_occlusion"], holdout=_HOLDOUT["o2_visual_occlusion"],
        notes="把「视觉代价」从「几何代价」里剥离出来：立板在 y 侧，不挡钳口走廊，"
              "所以失败只可能来自看不清，不会来自够不着。干扰块加一层「别乱碰」的约束。",
    ),
    "o3_front_overhang": dict(
        order=3, family="O", title="O3 前方低梁", xml="scenes/o3_front_overhang.xml",
        obstacle="目标 -x 侧、腕部高度的低梁（x 1.533~1.557，z 0.398~0.442，带两端立柱）",
        difficulty="**路径阻挡**：默认抓取走廊被压住，必须重新定向腕部绕过去",
        criterion="方块落到圆盘上 + 夹爪张开 + 已落在台面",
        sample_layout=layout_o3, success=lambda env: _on_target(env),
        settled=lambda env: _on_target(env), hold_s=0.4,
        instructions=_INSTR["o3_front_overhang"], holdout=_HOLDOUT["o3_front_overhang"],
        notes="**这一级是「给遥操作加腕部自由度」之后才做得出来的**：Uranus 抓取时腕部"
              "(link6_mount) 在齿面中点后约 12.4cm、高约 3cm、半径 3cm，默认姿态下要穿过"
              "x≈1.476 的 z≈0.37~0.43 空间，而低梁就在 1.533~1.557 -> 撞上。"
              "用 ↑/↓ 俯仰或 ←/→ 偏航把腕部挪开即可（限位实测 -50°~+55° / -50°~+75°）。",
    ),
    "o4_narrow_slot": dict(
        order=4, family="O", title="O4 窄缝插取", xml="scenes/o4_narrow_slot.xml",
        obstacle="两堵高 6cm 的导墙，缝宽 6.5cm（比目标顶还高 2cm，整个下探过程都受约束）",
        difficulty="**毫米级精度**：夹爪夹住 4cm 方块时齿面外廓 5.5cm -> 每侧只剩约 5mm",
        criterion="方块落到圆盘上 + 夹爪张开 + 已落在台面",
        sample_layout=layout_o4, success=lambda env: _on_target(env),
        settled=lambda env: _on_target(env), hold_s=0.4,
        instructions=_INSTR["o4_narrow_slot"], holdout=_HOLDOUT["o4_narrow_slot"],
        notes="缝宽是算出来的：齿面中心距 ±2.25cm + 齿厚半 0.51cm = 外廓 ±2.76cm（5.52cm），"
              "取 6.5cm -> 每侧 4.9mm。想调难度就改 o4 场景里槽壁的 pos.y 偏移。",
    ),
    "o5_covered_stack": dict(
        order=5, family="O", title="O5 覆盖堆叠", xml="scenes/o5_covered_stack.xml",
        obstacle="正压在目标上的另一块 4cm 方块（不同颜色）",
        difficulty="**长程 + 顺序**：不先搬开覆盖块就抓不到目标",
        criterion="方块落到圆盘上 + 夹爪张开 + 已落在台面",
        sample_layout=layout_o5, success=lambda env: _on_target(env),
        settled=lambda env: _on_target(env), hold_s=0.4,
        instructions=_INSTR["o5_covered_stack"], holdout=_HOLDOUT["o5_covered_stack"],
        notes="判据不额外硬编「必须先搬开」 —— 这是物理约束：覆盖块压着目标，"
              "直接合爪会把两块一起带起来。考的是长程记忆与子目标切换。",
    ),
}

LEVEL_ORDER = [k for k, _ in sorted(LEVELS.items(), key=lambda kv: kv[1]["order"])]


def get(level: str) -> dict:
    if level not in LEVELS:
        raise KeyError(f"未知等级 {level!r}；可用：{', '.join(LEVEL_ORDER)}")
    return LEVELS[level]


def describe_all() -> str:
    lines = [f"{'等级':<20} {'新增难度轴':<48} {'保持':>5}"]
    lines.append("-" * 80)
    for k in LEVEL_ORDER:
        s = LEVELS[k]
        lines.append(f"{s['title']:<20} {s['difficulty'][:46]:<48} {s['hold_s']:>4.1f}s")
    return "\n".join(lines)

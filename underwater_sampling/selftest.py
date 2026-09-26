# -*- coding: utf-8 -*-
"""场景自检：**这是替代"专家成功率标定"的验证手段**（本任务族按需求不提供专家策略）。

跑之前必须先确认场景是"人力可解"的，否则采出来的数据是在浪费人的时间。这里查五件事：

  1. 加载       —— 七个等级都能加载，底座需要的名字齐全，nq/qpos 顺序符合预期；
  2. 随机化     —— 抽 50 组布局：道具都在沉积物面之上、在台面范围内、互不重叠、离目标够远；
  3. 物理静置   —— 零控制下跑 0.4s：道具不下沉/不弹飞/不发散，L5 的管子保持竖直；
  4. IK 可达    —— 每级的关键位姿（物块上方/物块处/目标上方/目标处/工具架）解 IK 残差 < 3mm；
  5. 摆位碰撞   —— 把臂摆到关键位姿后做 forward，检查夹爪是否与静态道具（篮/井/立块/架）相撞。

另外打印两项**实测标定数据**（不改判据，只给调参用）：
  - L4 的夹持力：轻合 vs 一夹到底的峰值对比（用来定 force_limit）；
  - L6/L7 的针尖可达 x 范围（确认够得着测点）。

    python -m underwater_sampling.selftest             # 全部检查，退出码 0/1
    python -m underwater_sampling.selftest --levels l1_rock_collect l5_push_core
"""
from __future__ import annotations

import argparse
import sys
import traceback

import mujoco
import numpy as np

from .tasks import LEVEL_ORDER, SURFACE_Z, get
from .env import SamplingEnv

TOL_IK = 0.003            # IK 位置残差阈值（m）
SETTLE_STEPS = 200        # 0.4s @ 2ms


class Report:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.passed = 0
        self.failed: list[str] = []
        self.measured: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        flag = "PASS" if ok else "FAIL"
        if ok:
            self.passed += 1
        else:
            self.failed.append(f"{name} {detail}")
        if self.verbose:
            print(f"  [{flag}] {name}" + (f"   {detail}" if detail else ""))
        return bool(ok)

    def measure(self, text: str) -> None:
        self.measured.append(text)
        if self.verbose:
            print(f"  [实测] {text}")


# ----------------------------------------------------------------------
def _props_of(env):
    return dict(env.layout.get("props") or {})


def check_load(rep: Report, level: str, env: SamplingEnv) -> None:
    m = env.model
    # nq 下限取 8（臂 6 + 夹爪 2）：像 L8 这种被操作物是铰接件而不是自由体的等级，
    # 没有 freejoint，nq 就是 9，不能拿"必须 >= 15"去卡。
    rep.check(f"{level} 场景加载", m.nq >= 8 and m.nkey == 1,
              f"nq={m.nq} nv={m.nv} nkey={m.nkey}")
    need = ["object", "object_joint", "object_geom", "place_target", "place_target_site"]
    missing = [n for n in need
               if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) < 0
               and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) < 0
               and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n) < 0
               and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n) < 0]
    rep.check(f"{level} 底座命名齐全", not missing, f"缺 {missing}" if missing else "")
    # 夹爪标定约束：夹爪沿 y 合拢，而 env 用 geom_size[0](x 半宽) 标定 -> box 必须 x 尺寸 == y 尺寸；
    # 圆柱/胶囊的 x、y 延展都等于半径，天然满足（size[1] 对它们是"半长"，不是 y 半宽）。
    g = env.object_geom_id
    gtype = int(m.geom_type[g])
    sx = float(m.geom_size[g][0])
    if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        rep.check(f"{level} 目标物 x/y 尺寸相等（夹爪标定前提）",
                  abs(sx - float(m.geom_size[g][1])) < 1e-9,
                  f"box x={sx:.4f} y={float(m.geom_size[g][1]):.4f}")
    else:
        rep.check(f"{level} 目标物 x/y 延展相等（夹爪标定前提）", True,
                  f"type={gtype}（圆柱类）x=y=半径 {sx:.4f}")


def check_randomization(rep: Report, level: str, env: SamplingEnv, n: int = 50) -> None:
    rng = np.random.default_rng(12345)
    slab_x, slab_y = (1.72, 0.20), (-0.02, 0.22)     # 沉积物台面中心/半尺寸
    worst = {"on_surface": 1e9, "in_slab": 1e9, "min_pair": 1e9, "obj_goal": 1e9}
    for _ in range(n):
        lay = env.sample_layout(rng)
        pts = []
        for name, pose in (lay.get("props") or {}).items():
            p = np.asarray(pose[:3], dtype=float)
            pts.append((name, p))
            worst["on_surface"] = min(worst["on_surface"], p[2] - SURFACE_Z)
            worst["in_slab"] = min(worst["in_slab"],
                                   slab_x[1] - abs(p[0] - slab_x[0]),
                                   slab_y[1] - abs(p[1] - slab_y[0]))
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                worst["min_pair"] = min(worst["min_pair"],
                                        float(np.linalg.norm(pts[i][1][:2] - pts[j][1][:2])))
        # 有的等级没有 goal_xy（例如 L8 的目标是三维位置，走 statics 而不是地面标记）
        if lay.get("goal_xy") is not None:
            gx, gy = lay["goal_xy"]
            for _, p in pts:
                worst["obj_goal"] = min(worst["obj_goal"],
                                        float(np.hypot(p[0] - gx, p[1] - gy)))
    rep.check(f"{level} 道具都在沉积物面之上", worst["on_surface"] > -1e-6,
              f"最低余量 {worst['on_surface'] * 1000:.1f}mm")
    rep.check(f"{level} 道具都在台面范围内", worst["in_slab"] > 0.01,
              f"最近边距 {worst['in_slab'] * 1000:.0f}mm")
    if len(_props_of(env)) > 1:
        rep.check(f"{level} 道具互不重叠", worst["min_pair"] > 0.05,
                  f"最近间距 {worst['min_pair'] * 1000:.0f}mm")
    rep.check(f"{level} 目标物离目标点有距离", worst["obj_goal"] > 0.09,
              f"最近 {worst['obj_goal'] * 1000:.0f}mm")


def check_settle(rep: Report, level: str, env: SamplingEnv) -> None:
    """零控制跑 0.4s：道具应该原地静置（不下沉、不弹飞、不发散）。"""
    env.reset()
    z0 = {n: float(env.prop_pos(n)[2]) for n in _props_of(env)}
    env.data.ctrl[:] = np.concatenate([env.q_home, env._grip_ctrl(True)])
    for _ in range(SETTLE_STEPS):
        mujoco.mj_step(env.model, env.data)
    mujoco.mj_forward(env.model, env.data)
    drift = {n: float(env.prop_pos(n)[2] - z0[n]) for n in z0}
    bad = {n: d for n, d in drift.items() if abs(d) > 0.004 or not np.isfinite(d)}
    rep.check(f"{level} 静置不发散/不下沉", not bad,
              f"漂移 {({k: round(v * 1000, 1) for k, v in drift.items()})} mm")
    if level == "l5_push_core":
        tilt = env.object_tilt_deg()
        rep.check("l5_push_core 采样管静置后仍竖直", tilt < 5.0, f"倾角 {tilt:.2f}°")


def reach_targets(env: SamplingEnv) -> list[tuple[str, np.ndarray]]:
    """每级人必须够得到的关键位姿（齿面中点的目标位置）。"""
    obj = env.get_object_pos()
    goal = env.get_place_target()
    out = [("物块上方", obj + np.array([0, 0, 0.10])), ("物块处", obj.copy())]
    if env.level_name in ("l6_probe_touch", "l7_tool_roundtrip"):
        # 针尖在齿面中点 +x 方向伸出 8cm，所以齿面中点走到测点 -8cm 就能顶到
        out.append(("测点前方(针尖可及)", goal - np.array([0.08, 0, 0])))
    else:
        out += [("目标上方", goal + np.array([0, 0, 0.10])), ("目标处", goal.copy())]
    if env.level_name == "l7_tool_roundtrip":
        r = env.site_pos("tool_rack_site")
        out += [("工具架上方", r + np.array([0, 0, 0.10])), ("工具架处", r.copy())]
    return out


def check_reach(rep: Report, level: str, env: SamplingEnv) -> None:
    env.reset()
    q_home = env.q_home.copy()
    for label, target in reach_targets(env):
        q, err = env.solve_ik_grasp(np.asarray(target, float), q0=q_home, max_iter=500)
        e = float(np.linalg.norm(err))
        rep.check(f"{level} IK 可达：{label}", e < TOL_IK, f"残差 {e * 1000:.2f}mm")


def check_clearance(rep: Report, level: str, env: SamplingEnv) -> None:
    """把臂摆到关键位姿后 forward，检查夹爪有没有**真的咬进**静态道具。

    判据是**咬入深度**而不是"有没有接触点"：实测齿面长轴被张开角转了约 14°，端部会在
    0.1~0.2mm 量级贴住托盘侧壁/盘面，这种接触是接触生成的边界情形，不影响作业；
    而真正卡死的情况（link6_mount 啃 5cm 高的篮壁）咬入深度是 17~19mm，差两个数量级。
    所以阈值取 2mm，并把实测最大咬入深度打出来看余量。
    """
    PEN_TOL = 0.002
    env.reset()
    arm_geoms = set()
    for b in range(env.model.nbody):
        n = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if n.startswith(("jaw", "link", "base", "tcp")):
            arm_geoms |= env._body_geoms(b)
    static_geoms = set()
    for name in ("place_target", "seabed", "tool_rack", "marker_post", "object", "object2"):
        i = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i >= 0:
            static_geoms |= env._body_geoms(int(i))

    for label, target in reach_targets(env):
        q, _ = env.solve_ik_grasp(np.asarray(target, float), q0=env.q_home, max_iter=500)
        env.set_joints(q)
        mujoco.mj_forward(env.model, env.data)
        worst = 0.0
        pairs: list[str] = []
        for i in range(env.data.ncon):
            c = env.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if not ((g1 in arm_geoms and g2 in static_geoms)
                    or (g2 in arm_geoms and g1 in static_geoms)):
                continue
            pen = float(-c.dist)
            if pen > PEN_TOL:
                n1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g1)
                n2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, g2)
                pairs.append(f"{n1}↔{n2}({pen * 1000:.1f}mm)")
            worst = max(worst, pen)
        if worst > PEN_TOL or label in ("物块处", "目标处", "测点前方(针尖可及)", "工具架处"):
            rep.check(f"{label} 无咬入(>2mm)", worst <= PEN_TOL,
                      f"最大咬入 {worst * 1000:.1f}mm" + (f"  {sorted(set(pairs))[:3]}" if pairs else ""))
    env.reset()


# ----------------------------------------------------------------------
# 实测标定数据
# ----------------------------------------------------------------------
def measure_l4_force(rep: Report, env: SamplingEnv) -> None:
    """轻合 vs 一夹到底的峰值夹持力 —— **只作实测记录，不作判据**。

    实测结论：力峰值在 kN 量级且非单调（轻合 5220N / 标称 3801N / 到底 8139N），
    位置伺服压进硬接触的瞬态尖峰与 squeeze 不成正比，当"夹碎"阈值不可复现；
    加上现有遥操作是二值夹爪，操作者没有"更轻地合"这个动作，阈值也不可操作。
    所以 L4 的判据改成了"别碰倒"（见 tasks.success_l4）。
    """
    env.reset()
    out = {}
    for tag, angle in (("轻合(squeeze=2mm)", env.grasp_angle(squeeze=0.002)),
                       ("标称(squeeze=4mm)", env.grasp_angle(squeeze=0.004)),
                       ("一夹到底(jaw_min)", env.jaw_min)):
        env.reset()
        q, _ = env.solve_ik_grasp(env.get_object_pos(), q0=env.q_home, max_iter=500)
        peak = 0.0
        for _ in range(400):                       # 0.8s
            env.data.ctrl[:] = np.concatenate([q, [angle, angle]])
            mujoco.mj_step(env.model, env.data)
            peak = max(peak, env.force_grasp)
        out[tag] = peak
    rep.measure("L4 夹持力峰值（仅为记录，不作判据）: "
                + "  ".join(f"{k}={v:.1f}N" for k, v in out.items()))
    rep.measure("L4 判据 = 进托盘 + 静置 + 未翻倒（tilt<25°）；力阈值需模拟量夹爪控制才可用")


def measure_probe_reach(rep: Report, env: SamplingEnv) -> None:
    """针尖的 x 可达范围 vs 测点 x（确认够得着）。"""
    env.reset()
    goal = env.get_place_target()
    best = None
    for x in np.linspace(1.66, 1.78, 13):
        tgt = np.array([x - 0.08, goal[1], goal[2]])
        q, err = env.solve_ik_grasp(tgt, q0=env.q_home, max_iter=400)
        if float(np.linalg.norm(err)) < TOL_IK:
            best = x
        else:
            break
    rep.measure(f"{env.level_name} 针尖可达最大 x={best if best else float('nan'):.3f} "
                f"测点 x={goal[0]:.3f}  (针尖=齿面中点+0.08)")
    rep.check(f"{env.level_name} 针尖够得着测点", best is not None and best >= goal[0] - 1e-6)


def check_no_free_success(rep: Report, level: str, env: SamplingEnv) -> None:
    """开局绝不能已经判成功（否则数据全是白捡的）。"""
    env.reset()
    rep.check(f"{level} 开局不判成功", not env.success())
    m = env.metrics()
    rep.check(f"{level} metrics 字段齐全",
              all(k in m for k in ("level", "instruction", "force_grasp", "tilt_deg",
                                   "success", "settled")))
    rep.check(f"{level} 有语言指令", bool(m["instruction"]), repr(m["instruction"]))


# ----------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="采样任务族场景自检")
    ap.add_argument("--levels", nargs="*", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--no-physics", action="store_true", help="跳过物理静置与力标定（更快）")
    args = ap.parse_args(argv)

    levels = args.levels or LEVEL_ORDER
    rep = Report(verbose=not args.quiet)
    print(f"采样任务族自检：{len(levels)} 个等级\n" + "=" * 72)
    for level in levels:
        print(f"\n--- {get(level)['title']} ({level}) ---")
        try:
            env = SamplingEnv(level, seed=0)
        except Exception as e:
            rep.check(f"{level} 构造环境", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()
            continue
        check_load(rep, level, env)
        check_randomization(rep, level, env)
        check_no_free_success(rep, level, env)
        check_reach(rep, level, env)
        check_clearance(rep, level, env)
        if not args.no_physics:
            check_settle(rep, level, env)
        if level == "l4_fragile_coral" and not args.no_physics:
            measure_l4_force(rep, env)
        if level in ("l6_probe_touch", "l7_tool_roundtrip"):
            measure_probe_reach(rep, env)
        env.close() if hasattr(env, "close") else None

    print("\n" + "=" * 72)
    if rep.measured:
        print("实测标定数据：")
        for t in rep.measured:
            print("  · " + t)
    print(f"\n通过 {rep.passed} 项，失败 {len(rep.failed)} 项")
    if rep.failed:
        print("失败清单：")
        for f in rep.failed:
            print("  ✗ " + f)
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

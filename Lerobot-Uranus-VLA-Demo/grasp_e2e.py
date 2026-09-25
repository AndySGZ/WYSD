# -*- coding: utf-8 -*-
"""Uranus 抓取-放置 端到端验证。

两种用法：

  python grasp_e2e.py                 # 场景 keyframe 里的固定物块位置，打印每一步细节
  python grasp_e2e.py --randomize 20  # 台面上随机撒 20 组"物块位置 + 放置点"，统计成功率

专家逻辑在 src/expert_grasp.py（与 collect_data_grasp.py 共用同一套状态机）。

判据（全部满足才算通过）：
  1. 闭爪后抬升时物块跟着上升（最高点比初始高 >= 8cm）；
  2. 物块最终落在放置点附近（水平 < 5cm）；
  3. env.success() 为 True。
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env_grasp import UranusGraspEnv
from src.expert_grasp import pick_place


def run_single(args) -> int:
    env = UranusGraspEnv(seed=0)
    env.reset()
    print(f"物块尺寸: {env.object_width * 100:.1f} cm   mass={env.model.body_mass[env.object_body_id]:.3f} kg")
    print(f"张开状态净间距 {env.measured_clearance() * 1000:.1f} mm"
          f" | 夹爪闭合角={env.grasp_angle():.3f} rad -> 净间距 {env.tooth_clearance(env.grasp_angle()) * 1000:.1f} mm")

    res = pick_place(env, collect_log=True)

    print()
    print(f"{'step':>12} | {'object xyz':>22} | {'tooth-mid xyz':>22} | {'gap':>7} | {'jaw':>6}")
    print("-" * 84)
    for tag, obj, tm, gap, grip in res["log"]:
        print(f"{tag:>12} | {np.round(obj, 3)!s:>22} | {np.round(tm, 3)!s:>22} | "
              f"{gap * 1000:5.1f}mm | {grip:+.3f}")

    print()
    print(f"最终物块 xyz      = {np.round(res['obj_end'], 4)}")
    print(f"放置点 xyz        = {np.round(res['place'], 4)}")
    print(f"抬升高度          = {res['lift'] * 1000:+.1f} mm   (需 > 80mm)")
    print(f"放置水平误差      = {res['horiz_err'] * 1000:.1f} mm    (需 < 50mm)")
    print(f"env.success()     = {res['success']}")
    print("RESULT:", "PASS" if res["ok"] else "FAIL")
    return 0 if res["ok"] else 1


def run_randomize(args) -> int:
    env = UranusGraspEnv(seed=0)
    rng = np.random.default_rng(args.seed)
    print(f"随机化回归: {args.randomize} 组 (seed={args.seed})")
    print(f"{'#':>4} | {'pick xy':>16} | {'place xy':>16} | {'lift':>9} | {'horiz err':>10} | {'result':>6}")
    print("-" * 78)

    n_ok = 0
    lifts, errs = [], []
    for i in range(args.randomize):
        obj_pose, place = env.sample_layout(rng)
        res = pick_place(env, obj_pose=obj_pose, place_xyz=place)
        lifts.append(res["lift"])
        errs.append(res["horiz_err"])
        n_ok += int(res["ok"])
        if not res["ok"] or args.verbose:
            print(f"{i:>4} | {np.round(obj_pose[:2], 3)!s:>16} | {np.round(place[:2], 3)!s:>16} | "
                  f"{res['lift'] * 1000:+8.1f}mm | {res['horiz_err'] * 1000:9.1f}mm | "
                  f"{'PASS' if res['ok'] else 'FAIL':>6}")

    print("-" * 78)
    print(f"expert success: {n_ok}/{args.randomize} = {n_ok / args.randomize:.1%}")
    print(f"lift    mean={np.mean(lifts) * 1000:.1f}mm  min={np.min(lifts) * 1000:.1f}mm")
    print(f"plc err mean={np.mean(errs) * 1000:.1f}mm  max={np.max(errs) * 1000:.1f}mm")
    return 0 if n_ok == args.randomize else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomize", type=int, default=0,
                        help="随机化回归的 episode 数（0=只跑固定位置的详细流程）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true", help="随机化时逐条打印成功 case")
    args = parser.parse_args()
    if args.randomize > 0:
        return run_randomize(args)
    return run_single(args)


if __name__ == "__main__":
    sys.exit(main())

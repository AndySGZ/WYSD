# -*- coding: utf-8 -*-
"""抓取任务的数据采集：用 src/expert_grasp.py 的脚本专家生成 (obs, action) 示范。

与教程 `collect_data.py`（遥操作采集）对应；reach 版用"单个 IK 解"当专家，
这里因为要让物块真的被抓起来、跟着走，专家必须跑接触动力学，所以录的是**完整轨迹**：
每个 episode 在台面上随机一个"物块位置 + 放置点"，跑一遍
approach -> descend -> close -> lift -> transport -> place_down -> release，
按控制周期（默认每 25 个物理步 = 50ms）录一帧。

    obs    = [关节(6), 物块xyz(3), 物块四元数(4), 放置点xyz(3), 夹爪(1)]   = 17 维
    action = 未来第 K 拍控制周期的实测 [关节角(6), 夹爪角(1)]              = 7 维

动作语义：reach 存"绝对目标关节角"，这里存"K 拍(默认0.2s)之后的实测关节角/夹爪角"。
原因见 src/expert_grasp.py 文件头：位置伺服跟得太紧，直接学"当前目标"会退化成恒等映射、
闭环卡死。夹爪是连续关节角（≈0.208 张开 / ≈0.135 夹住物块），"慢慢张开"的斜坡也在数据里。

脚本会把专家自己跑失败的 episode 丢掉（默认），只把成功示范写进数据集。

用法：
    python collect_data_grasp.py --num_episodes 60
    python collect_data_grasp.py --num_episodes 200 --rec_every 25 --lookahead 4 --out_dir data/grasp
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env_grasp import UranusGraspEnv
from src.expert_grasp import pick_place


def main(args):
    env = UranusGraspEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    obs_list, act_list, ep_ends = [], [], []
    n_ok = 0
    n_try = 0
    t0 = time.time()

    while n_ok < args.num_episodes and n_try < args.max_tries:
        n_try += 1
        obj_pose, place = env.sample_layout(rng)
        res = pick_place(env, obj_pose=obj_pose, place_xyz=place,
                         record=True, rec_every=args.rec_every, lookahead=args.lookahead,
                         squeeze=args.squeeze)
        if not res["ok"]:
            if args.keep_failures:
                pass
            else:
                print(f"  [{n_try}] 专家失败，丢弃  pick={np.round(obj_pose[:2], 3)} "
                      f"place={np.round(place[:2], 3)} lift={res['lift'] * 1000:.0f}mm "
                      f"err={res['horiz_err'] * 1000:.0f}mm")
                continue
        if not res["frames"]:
            print(f"  [{n_try}] 没录到帧，跳过")
            continue

        for obs, act in res["frames"]:
            obs_list.append(obs)
            act_list.append(act)
        ep_ends.append(len(obs_list))
        n_ok += 1
        if n_ok % args.print_every == 0 or n_ok == args.num_episodes:
            print(f"  collected {n_ok}/{args.num_episodes} episodes "
                  f"(try={n_try}, frames={len(obs_list)}, "
                  f"lift={res['lift'] * 1000:.0f}mm, err={res['horiz_err'] * 1000:.1f}mm, "
                  f"{time.time() - t0:.0f}s)")

    if not obs_list:
        print("没采到任何数据")
        return 1

    observations = np.stack(obs_list, axis=0).astype(np.float32)
    actions = np.stack(act_list, axis=0).astype(np.float32)
    episode_ends = np.array(ep_ends, dtype=np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "data.npz",
             observations=observations,
             actions=actions,
             episode_ends=episode_ends)
    print()
    print(f"Saved {observations.shape[0]} frames / {len(episode_ends)} episodes "
          f"-> {out_dir / 'data.npz'}")
    print(f"  observations={observations.shape}  actions={actions.shape}")
    print(f"  专家成功率 {n_ok}/{n_try} = {n_ok / max(n_try, 1):.1%}，"
          f"平均每 episode {observations.shape[0] / len(episode_ends):.0f} 帧，"
          f"耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_episodes", type=int, default=50, help="要采多少个成功 episode")
    parser.add_argument("--max_tries", type=int, default=1000, help="最多尝试多少次（含失败）")
    parser.add_argument("--out_dir", type=str, default="data/grasp")
    parser.add_argument("--rec_every", type=int, default=25,
                        help="每多少个物理步(2ms)录一帧，25 -> 50ms -> 20Hz")
    parser.add_argument("--lookahead", type=int, default=4,
                        help="动作取未来第几拍的状态（1 拍 = rec_every 个物理步 = 50ms）")
    parser.add_argument("--squeeze", type=float, default=0.004, help="夹持压入量(m)")
    parser.add_argument("--keep_failures", action="store_true",
                        help="把专家失败的 episode 也写进数据集（用于对比，默认丢弃）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    sys.exit(main(parser.parse_args()))

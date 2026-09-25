"""脚本化专家数据采集：用 IK 作为专家，生成"点到点到达"示范。

与教程 `collect_data.py` 的"遥操作 + LeRobotDataset"对应，这里因为 Uranus
任务是简单的到达任务，用 IK 作为专家，生成 (obs, action) 数据并保存为 .npz。

obs    = [joint1..joint6(当前), goal_x, goal_y, goal_z]        (9)
action = [joint1..joint6(目标关节角 = IK(goal)), gripper(恒开=0.0)]  (7)

说明：action 采用"绝对目标关节角"（IK 解），而不是"下一帧关节角"。
这样策略学的是一个 goal -> 关节目标 的映射（神经 IK）。若 action 用"下一帧关节角"
做闭环轨迹跟踪，微小误差会在闭环中逐步累积（behavior cloning 的分布漂移问题），
成功率会明显下降；用绝对目标则每一步都朝正确目标走，稳健得多。
为让策略对"任意当前构型"都稳健，obs 里的当前关节角仍沿 home->target 轨迹遍历，
但 action 始终是同一个 IK 目标。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env import UranusReachEnv


def linear_interp(q0, q1, steps):
    """关节空间线性插值：每个 step 大小均匀，避免 min-jerk 起步过慢导致
    专家动作几乎为 0、被策略噪声淹没的问题。"""
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    s = np.linspace(0.0, 1.0, steps)
    return q0[None, :] + s[:, None] * (q1 - q0)[None, :]


def main(args):
    cfg_path = Path(args.config)
    cfg = json.loads(cfg_path.read_text())

    env = UranusReachEnv(
        xml_file=cfg["xml_file"],
        goal_r=tuple(cfg["goal"]["r_range"]),
        goal_theta_deg=tuple(cfg["goal"]["theta_deg_range"]),
        goal_z=tuple(cfg["goal"]["z_range"]),
        success_thresh=cfg["success_thresh"],
        seed=args.seed,
    )

    obs_list, act_list, ep_ends = [], [], []
    gripper = 0.0  # 本任务夹爪恒开（-0.2 对应开，0 表示"开"的命令标签）

    for ep in range(args.num_episodes):
        obs = env.reset()
        goal = env.goal.copy()
        q_target, ik_err = env.solve_ik(goal)
        if np.linalg.norm(ik_err) > 0.02:
            print(f"  episode {ep}: IK failed (err={np.linalg.norm(ik_err):.4f}), skip")
            continue
        env.set_joints(env.q_home)  # solve_ik 会移动臂，这里复位回 home 再开始记录

        # 加入一段随机中间姿态，增加示范多样性（可选，默认关闭）
        via = None
        if args.via_ratio > 0.0 and env.rng.random() < args.via_ratio:
            via = q_target + env.rng.normal(0.0, 0.05, size=len(q_target))
            via = np.clip(via, env.q_min, env.q_max)

        # 生成轨迹 home -> (via) -> target，末尾停留若干帧，教会策略"到达后保持"
        waypoints = [env.q_home] + ([via] if via is not None else []) + [q_target]
        total_steps = args.episode_len
        hold = max(0, args.hold_frames - 1)  # 目标处额外保持帧数
        if via is not None:
            # 有 via 时：分段线性插值
            seg_steps = max(1, (total_steps - hold) // (len(waypoints) - 1))
            traj = []
            for i in range(len(waypoints) - 1):
                traj.append(linear_interp(waypoints[i], waypoints[i + 1], seg_steps))
            traj = np.concatenate(traj, axis=0)
        else:
            move_steps = max(1, total_steps - hold)
            traj = linear_interp(env.q_home, q_target, move_steps)
        if hold > 0:
            traj = np.concatenate([traj, np.tile(q_target, (hold, 1))], axis=0)
        traj = traj[: total_steps]  # 保证恰好 total_steps 个点

        for t in range(total_steps - 1):
            q_now = env.get_joints()
            obs_list.append(np.concatenate([q_now, goal]).astype(np.float32))
            act = np.concatenate([q_target, [gripper]]).astype(np.float32)  # 绝对目标关节角
            act_list.append(act)
            env.set_joints(traj[t + 1])  # 仅用于遍历"当前构型"以增加 obs 覆盖

        ep_ends.append(len(obs_list))
        if (ep + 1) % args.print_every == 0:
            print(f"  collected {ep + 1}/{args.num_episodes} episodes, frames={len(obs_list)}")

    observations = np.stack(obs_list, axis=0)
    actions = np.stack(act_list, axis=0)
    episode_ends = np.array(ep_ends, dtype=np.int64)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "data.npz",
        observations=observations,
        actions=actions,
        episode_ends=episode_ends,
    )
    print(f"Saved {observations.shape[0]} frames / {len(episode_ends)} episodes -> {out_dir / 'data.npz'}")
    print(f"  obs shape={observations.shape}, action shape={actions.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task.json")
    parser.add_argument("--out_dir", type=str, default="data/reach")
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--episode_len", type=int, default=100)
    parser.add_argument("--hold_frames", type=int, default=10)
    parser.add_argument("--via_ratio", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=20)
    args = parser.parse_args()
    main(args)

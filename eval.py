"""评测：加载训练好的策略，在随机目标上 rollout，统计成功率。

与教程 `12.eval_custom.ipynb` 对应。每个 episode：reset -> 采样目标 ->
策略闭环输出目标关节角 -> env.step -> 判断 TCP 是否到达目标。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env import UranusReachEnv
from train import ReachMLP


def main(args):
    cfg = json.loads(Path(args.config).read_text())
    env = UranusReachEnv(
        xml_file=cfg["xml_file"],
        goal_r=tuple(cfg["goal"]["r_range"]),
        goal_theta_deg=tuple(cfg["goal"]["theta_deg_range"]),
        goal_z=tuple(cfg["goal"]["z_range"]),
        success_thresh=cfg["success_thresh"],
        seed=args.seed,
    )

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = ReachMLP(ckpt["in_dim"], ckpt["out_dim"], ckpt["hidden_dim"], ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    obs_mean = torch.from_numpy(ckpt["obs_mean"])
    obs_std = torch.from_numpy(ckpt["obs_std"])
    act_mean = torch.from_numpy(ckpt["act_mean"])
    act_std = torch.from_numpy(ckpt["act_std"])

    successes = 0
    dists = []
    for ep in range(args.num_episodes):
        obs = env.reset()
        done = False
        ep_dist = None
        for t in range(args.horizon):
            x = torch.from_numpy(obs.astype(np.float32))
            x = (x - obs_mean) / obs_std
            with torch.no_grad():
                act_n = model(x.unsqueeze(0)).squeeze(0)
            act = act_n * act_std + act_mean
            act = act.numpy()
            obs, info = env.step(act)
            ep_dist = info["dist"]
            if info["success"]:
                done = True
                break
        if done:
            successes += 1
        dists.append(ep_dist)
        if args.verbose:
            print(f"  ep {ep}: dist={ep_dist:.3f} success={done}")

    rate = successes / args.num_episodes
    print(f"\nSuccess rate: {successes}/{args.num_episodes} = {rate:.2%}")
    print(f"Final TCP-goal dist: mean={np.mean(dists):.3f}, median={np.median(dists):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/task.json")
    parser.add_argument("--ckpt_path", type=str, default="ckpt/reach/policy.pt")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args)

"""行为克隆训练：MLP 从 (关节角, 目标) 回归目标关节角。

与教程 `train_custom.py` 的 LeRobot BaselinePolicy 对应，这里用一个纯 torch MLP，
输入 obs(9) 输出 action(7)，MSE 损失。数据量小、无图像，CPU 即可训练。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))


class ReachMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=256, n_layers=4):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main(args):
    data = np.load(args.data_path)
    obs = torch.from_numpy(data["observations"]).float()
    act = torch.from_numpy(data["actions"]).float()
    print(f"dataset: obs={tuple(obs.shape)}, action={tuple(act.shape)}")

    # 归一化统计
    # 注意 std 的下限不能取 1e-6 这种极小值：obs 里有些维度几乎是常数（例如放置点 z
    # 恒为 0.375），一旦下限太小，(x-mean)/std 会在状态稍微偏离时爆成成百上千，
    # 网络输出随之饱和到关节限位——实测策略在抓住物块后突然把手臂甩飞。
    std_floor = 1e-2
    obs_mean = obs.mean(0)
    obs_std = obs.std(0).clamp_min(std_floor)
    act_mean = act.mean(0)
    act_std = act.std(0).clamp_min(std_floor)

    def norm(x, m, s):
        return (x - m) / s

    obs_n = norm(obs, obs_mean, obs_std)
    act_n = norm(act, act_mean, act_std)

    n = obs.shape[0]
    n_train = int(n * (1.0 - args.val_ratio))
    idx = torch.randperm(n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]

    model = ReachMLP(obs.shape[1], act.shape[1], args.hidden_dim, args.n_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    tr_obs, tr_act = obs_n[tr_idx], act_n[tr_idx]
    va_obs, va_act = obs_n[va_idx], act_n[va_idx]

    best_va = float("inf")
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train)
        total = 0.0
        for i in range(0, n_train, args.batch_size):
            b = perm[i:i + args.batch_size]
            xb, yb = tr_obs[b], tr_act[b]
            yb_hat = model(xb)
            loss = loss_fn(yb_hat, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(b)
        train_loss = total / n_train

        model.eval()
        with torch.no_grad():
            va_loss = loss_fn(model(va_obs), va_act).item()

        if va_loss < best_va:
            best_va = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % args.print_every == 0:
            print(f"epoch {epoch + 1:4d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={va_loss:.6f}")

    ckpt_dir = Path(args.ckpt_path)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "obs_mean": obs_mean.numpy(),
            "obs_std": obs_std.numpy(),
            "act_mean": act_mean.numpy(),
            "act_std": act_std.numpy(),
            # 演示数据的动作范围：评测时把策略输出 clip 进来，避免外推出界
            # （BC 策略一旦偏离演示分布，MLP 会输出饱和的离谱动作，把机器人甩飞）
            "act_min": act.min(0).values.numpy(),
            "act_max": act.max(0).values.numpy(),
            "in_dim": obs.shape[1],
            "out_dim": act.shape[1],
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
        },
        ckpt_dir / "policy.pt",
    )
    print(f"best val loss={best_va:.6f}, saved -> {ckpt_dir / 'policy.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/reach/data.npz")
    parser.add_argument("--ckpt_path", type=str, default="ckpt/reach")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--print_every", type=int, default=10)
    args = parser.parse_args()
    main(args)

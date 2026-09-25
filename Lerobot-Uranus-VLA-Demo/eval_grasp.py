# -*- coding: utf-8 -*-
"""抓取策略评测：加载动作块策略，在随机"物块位置 + 放置点"上闭环 rollout，统计成功率。

与教程 `12.eval_custom.ipynb` 对应。每个 episode：
    reset(随机布局) -> 循环 { 策略(obs) 输出 H 步动作块 -> 依次执行 H 个控制周期 }
                    -> env.success()

action = 未来 K 拍（默认 0.2s）之后的 [关节角(6), 夹爪角(1)]。执行时把它当作接下来
K 拍的目标**保持住**，每拍跑 `--ctrl_steps` 个物理步（默认 25 步 = 50ms），K 拍后再重新推理。

为什么要保持 K 拍而不是每拍都重推：动作本身就是"K 拍之后的状态"，如果每 50ms 都拿它当目标，
等于让手臂用 K 倍速度去追一个移动目标，实测在每段运动末尾会冲过头（齿面冲到物块中心下方
4cm）。保持 K 拍后手臂按专家原本的速度走，且下一次推理时状态正好落回演示分布内。

判据与专家一致（见 src/expert_grasp.py）：抬升 >= 8cm、放置水平误差 < 5cm、success()。

用法：
    python eval_grasp.py --ckpt_path ckpt/grasp/policy.pt --num_episodes 50 --verbose
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.env_grasp import UranusGraspEnv
from src.expert_grasp import MIN_LIFT
from train import ReachMLP

import mujoco


def main(args):
    env = UranusGraspEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    model = ReachMLP(ckpt["in_dim"], ckpt["out_dim"], ckpt["hidden_dim"], ckpt["n_layers"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    obs_mean = torch.from_numpy(ckpt["obs_mean"])
    obs_std = torch.from_numpy(ckpt["obs_std"])
    act_mean = torch.from_numpy(ckpt["act_mean"])
    act_std = torch.from_numpy(ckpt["act_std"])
    act_min = ckpt.get("act_min")
    act_max = ckpt.get("act_max")
    print(f"policy: obs_dim={ckpt['in_dim']} action_dim={ckpt['out_dim']}"
          f"{' (动作 clip 到演示范围)' if act_min is not None else ''}")

    n_ok = 0
    lifts, errs = [], []
    for ep in range(args.num_episodes):
        obj_pose, place = env.sample_layout(rng)
        env.reset(object_pose=obj_pose)
        obs0 = env.get_object_pos().copy()
        peaks = [obs0[2]]

        # 每 hold 拍推理一次，其间保持同一个目标
        for _ in range(int(np.ceil(args.horizon / args.hold))):
            x = torch.from_numpy(env.get_obs().astype(np.float32))
            x = (x - obs_mean) / obs_std
            with torch.no_grad():
                act = model(x.unsqueeze(0)).squeeze(0)
            act = (act * act_std + act_mean).numpy()
            if act_min is not None:
                act = np.clip(act, act_min, act_max)

            q = np.clip(act[:6], env.q_min, env.q_max)
            jaw = float(np.clip(act[6], env.jaw_min, env.jaw_max))
            env.data.ctrl[:] = np.concatenate([q, [jaw, jaw]])
            for _ in range(args.hold * args.ctrl_steps):
                mujoco.mj_step(env.model, env.data)
                peaks.append(env.get_object_pos()[2])

        obj_end = env.get_object_pos()
        lift = float(max(peaks) - obs0[2])
        err = float(np.linalg.norm(obj_end[:2] - place[:2]))
        ok = bool(lift > MIN_LIFT and env.success())
        lifts.append(lift)
        errs.append(err)
        n_ok += int(ok)
        if args.verbose or not ok:
            print(f"  ep {ep:3d}: pick={np.round(obj_pose[:2], 3)} place={np.round(place[:2], 3)} "
                  f"lift={lift * 1000:+7.1f}mm err={err * 1000:7.1f}mm {'PASS' if ok else 'FAIL'}")

    print()
    print(f"Success rate: {n_ok}/{args.num_episodes} = {n_ok / args.num_episodes:.2%}")
    print(f"lift    mean={np.mean(lifts) * 1000:.1f}mm  min={np.min(lifts) * 1000:.1f}mm")
    print(f"plc err mean={np.mean(errs) * 1000:.1f}mm  max={np.max(errs) * 1000:.1f}mm")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="ckpt/grasp/policy.pt")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=180,
                        help="控制周期数上限（专家一个 episode 约 150 拍）")
    parser.add_argument("--hold", type=int, default=4,
                        help="每次推理后保持同一个目标的控制周期数（应与采集的 lookahead 一致）")
    parser.add_argument("--ctrl_steps", type=int, default=25,
                        help="每个控制周期执行的物理步数（与采集的 rec_every 一致）")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    sys.exit(main(parser.parse_args()))

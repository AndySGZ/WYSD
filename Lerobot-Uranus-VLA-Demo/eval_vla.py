# -*- coding: utf-8 -*-
"""用 LeRobot 训练/微调好的 VLA 策略（pi0 / SmolVLA / ACT）在 Uranus 抓取环境里闭环评测。

    obs   : observation.images.top / observation.images.wrist  (3,H,W)
            observation.state (7,) = 关节1..6 + 夹爪角
            task  = 语言指令
    执行  : 每个决策点推理一次，把动作块的前 n 步依次下发（与采集的 fps=20 对齐）
    判据  : 与仓库其它脚本一致 —— 抬升 > MIN_LIFT(2cm) 且 env.success()

依赖：必须在**装了 lerobot 策略依赖的环境**里跑（见 VLA_FINETUNE.md）。
本仓库的 lerobot_mujoco_vla_tutorial 环境目前缺 transformers、huggingface_hub 版本过旧，
`from lerobot.policies.factory import ...` 会直接失败 —— 脚本会明确告诉你。

本地自检（不需要 lerobot 策略，只验证评测回路本身）：
    python eval_vla.py --dummy_mlp ckpt/manual/policy.pt --num_episodes 3

真跑 VLA：
    python eval_vla.py --ckpt outputs/train/uranus_smolvla/checkpoints/last/pretrained_model \
                       --num_episodes 50 --images top,wrist
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import torch

from src.env_grasp import UranusGraspEnv
from src.expert_grasp import MIN_LIFT

TASK = "put the yellow cube on the blue disc"


# ----------------------------------------------------------------------
# 策略接口：都集中在这里，换 lerobot 版本时只改这一段
# ----------------------------------------------------------------------
def load_policy(ckpt_path: str):
    """加载 LeRobot 策略 + 前后处理。返回 (policy, preprocess, postprocess, image_keys)。"""
    try:
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    except ImportError as e:
        raise SystemExit(
            f"！这个环境装不了/没装 LeRobot 策略依赖：{e}\n"
            f"  π0 / SmolVLA 的微调与评测环境怎么建，见 VLA_FINETUNE.md\n"
            f"  只想验证评测回路，可用：--dummy_mlp ckpt/manual/policy.pt"
        )

    cfg = PreTrainedConfig.from_pretrained(ckpt_path)
    cfg.pretrained_path = Path(ckpt_path)
    policy = get_policy_class(cfg.type).from_pretrained(ckpt_path, config=cfg)
    policy.eval()
    # 归一化统计随 checkpoint 存；不同版本参数名略有差异，逐个试
    pre = post = None
    for kwargs in ({"pretrained_path": Path(ckpt_path)}, {}, None):
        try:
            pre, post = make_pre_post_processors(cfg, **(kwargs or {}))
            break
        except Exception:
            continue
    if pre is None:
        print("！没能构造 pre/post processor（版本差异），动作将按原样使用。")
    print(f"policy: type={cfg.type}  ckpt={ckpt_path}")
    return policy, pre, post


def policy_action(policy, pre, post, obs_batch):
    """一次推理，返回 (n, 7) numpy 动作块。"""
    with torch.no_grad():
        x = pre(obs_batch) if pre is not None else obs_batch
        if hasattr(policy, "predict_action_chunk"):
            act = policy.predict_action_chunk(x)
        else:
            act = policy.select_action(x).unsqueeze(0)
        if post is not None:
            act = post(act)
    return act.squeeze(0).cpu().numpy()


# ----------------------------------------------------------------------
# 观测构造（VLA 和 dummy 两种模式共用同一个回路）
# ----------------------------------------------------------------------
class ObsBuilder:
    def __init__(self, env, size, cams, device):
        self.env, self.cams, self.device = env, cams, device
        self.renderer = mujoco.Renderer(env.model, size, size)

    def images(self):
        out = {}
        for key, cam in self.cams:
            self.renderer.update_scene(self.env.data, camera=cam)
            img = self.renderer.render()                       # (H,W,3) uint8
            out[key] = torch.from_numpy(img).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        return out

    def build(self):
        d = self.images()
        d["observation.state"] = torch.from_numpy(
            np.concatenate([self.env.get_joints(), [self.env.get_gripper()]]).astype(np.float32)
        ).unsqueeze(0)
        d["task"] = [TASK]
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in d.items()}

    def close(self):
        self.renderer.close()


def main(args):
    env = UranusGraspEnv(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  episodes={args.num_episodes}  env=camera 图像 {args.image_size}")

    cam_pairs = []
    for name in args.images.split(","):
        name = name.strip()
        cam_pairs.append((f"observation.images.{name}", args.top_cam if name == "top" else args.wrist_cam))
    obsb = ObsBuilder(env, args.image_size, cam_pairs, device)

    policy = pre = post = None
    mlp = None
    if args.dummy_mlp:
        from train import ReachMLP
        ck = torch.load(args.dummy_mlp, map_location="cpu", weights_only=False)
        mlp = ReachMLP(ck["in_dim"], ck["out_dim"], ck["hidden_dim"], ck["n_layers"])
        mlp.load_state_dict(ck["state_dict"]); mlp.eval()
        print(f"[dummy] 用本仓库 MLP 验证评测回路（不看图像）：{args.dummy_mlp}")
    else:
        policy, pre, post = load_policy(args.ckpt)

    n_ok = 0
    for ep in range(args.num_episodes):
        obj_pose, place = env.sample_layout(rng)
        env.reset(object_pose=obj_pose)
        obj0 = env.get_object_pos().copy()
        peaks = [obj0[2]]

        for t in range(args.horizon):
            obs = obsb.build()
            if mlp is not None:
                x = torch.from_numpy(env.get_obs().astype(np.float32))
                with torch.no_grad():
                    act = mlp(x.unsqueeze(0)).squeeze(0).numpy().reshape(1, 7)
            else:
                act = policy_action(policy, pre, post, obs)

            for k in range(min(args.n_action_steps, act.shape[0])):
                q = np.clip(act[k, :6], env.q_min, env.q_max)
                jaw = float(np.clip(act[k, 6], env.jaw_min, env.jaw_max))
                env.data.ctrl[:] = np.concatenate([q, [jaw, jaw]])
                for _ in range(args.ctrl_steps):
                    mujoco.mj_step(env.model, env.data)
                peaks.append(env.get_object_pos()[2])

        lift = float(max(peaks) - obj0[2])
        err = float(np.linalg.norm(env.get_object_pos()[:2] - place[:2]))
        ok = bool(lift > MIN_LIFT and env.success())
        n_ok += int(ok)
        if args.verbose or not ok:
            print(f"  ep {ep:3d}: lift={lift * 1000:+7.1f}mm err={err * 1000:7.1f}mm "
                  f"{'PASS' if ok else 'FAIL'}")
        if ep % 10 == 9:
            print(f"  ...{ep + 1}/{args.num_episodes} 已跑，当前成功率 {n_ok}/{ep + 1}")

    obsb.close()
    print(f"\nSuccess rate: {n_ok}/{args.num_episodes} = {n_ok / args.num_episodes:.2%}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None, help="LeRobot 策略 checkpoint 目录或 HF repo")
    parser.add_argument("--dummy_mlp", type=str, default=None,
                        help="用本仓库 MLP 走一遍评测回路（不需要 lerobot 策略，用于自检）")
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=40, help="决策点上限（每个决策点跑 n_action_steps 拍）")
    parser.add_argument("--n_action_steps", type=int, default=8, help="每个决策点执行动作块的前几步")
    parser.add_argument("--ctrl_steps", type=int, default=25, help="每个控制周期的物理步数（25*2ms=50ms=20Hz）")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--images", type=str, default="top,wrist", help="用到哪几路相机")
    parser.add_argument("--top_cam", type=str, default="grasp_cam")
    parser.add_argument("--wrist_cam", type=str, default="wrist_cam")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    sys.exit(main(parser.parse_args()))

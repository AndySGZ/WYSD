"""Uranus reach 环境的冒烟测试：验证模型加载、IK、运动学/动力学 step。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from src.env import UranusReachEnv


def main():
    env = UranusReachEnv(seed=0)
    print(f"[1] model loaded: nq={env.model.nq}, nu={env.model.nu}")
    print(f"    arm_qpos_idx={env.arm_qpos_idx.tolist()}")
    print(f"    q_home={np.round(env.q_home, 4).tolist()}")

    # reset 与目标采样
    obs = env.reset()
    goal = env.goal.copy()
    print(f"[2] reset ok, goal={np.round(goal, 3).tolist()}, obs_dim={obs.shape[0]}")

    # IK
    q, err = env.solve_ik(goal)
    print(f"[3] IK err={np.linalg.norm(err):.2e}, tcp={np.round(env.get_tcp_pos(), 3).tolist()}")

    # 运动学 step
    _, info = env.step(np.concatenate([q, [0.0]]))
    assert info["dist"] < 1e-2, f"kinematic step dist too large: {info['dist']}"
    print(f"[4] kinematic step dist={info['dist']:.4f}, success={info['success']}")

    # 动力学 step（串联简化：6 关节 + 1 夹爪直接映射；重力下会有 ~3-5cm 稳态下垂）
    env.reset(goal=goal)
    _, info_dyn = env.step_dynamics(np.concatenate([q, [0.0]]), substeps=3000)
    assert info_dyn["dist"] < 0.12, f"dynamics step dist too large: {info_dyn['dist']}"
    print(f"[5] dynamics step dist={info_dyn['dist']:.4f}, tcp={np.round(info_dyn['tcp'], 3).tolist()}")

    # 目标可采样性
    goals = np.array([env.sample_goal() for _ in range(20)])
    print(f"[6] sampled 20 goals: r range=[{np.linalg.norm(goals[:, :2], axis=1).min():.2f}, "
          f"{np.linalg.norm(goals[:, :2], axis=1).max():.2f}], z range=[{goals[:, 2].min():.2f}, {goals[:, 2].max():.2f}]")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()

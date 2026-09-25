# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

obj = np.array([1.4, 0.12, 0.36])

def run(kp_scale):
    env = UranusGraspEnv(seed=0)
    q, err = env.solve_ik_grasp(obj)
    arm_act = np.array([mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j+'_act') for j in env.ARM_JOINTS])
    base_kp = env.model.actuator_gainprm[arm_act, 0].copy()
    base_kv = -env.model.actuator_biasprm[arm_act, 2].copy()
    kp = base_kp * kp_scale
    kv = base_kv * np.sqrt(kp_scale)  # scale kv by sqrt to keep damping ratio
    env.model.actuator_gainprm[arm_act, 0] = kp
    env.model.actuator_biasprm[arm_act, 1] = -kp
    env.model.actuator_biasprm[arm_act, 2] = -kv

    env.set_joints(env.q_home)
    env.set_gripper(open=True)
    for _ in range(3000):
        env.data.ctrl[:] = np.concatenate([q, env.GRIPPER_OPEN])
        mujoco.mj_step(env.model, env.data)
    tm = env.tooth_midpoint()
    err = obj - tm
    # stability check: qvel magnitude
    qvel = np.abs(env.data.qvel[env.arm_dofadr]).max()
    print(f"kp_scale={kp_scale:6.1f}  err={np.round(err,4)}  |err|={np.linalg.norm(err)*1000:.1f}mm  max_qvel={qvel:.3e}")
    return np.linalg.norm(err)

for s in [1, 5, 10, 20, 50, 100]:
    try:
        run(s)
    except Exception as e:
        print(f"kp_scale={s}: FAILED {e}")

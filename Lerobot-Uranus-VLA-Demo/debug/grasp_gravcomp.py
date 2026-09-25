# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
obj = np.array([1.4, 0.12, 0.36])
q, err = env.solve_ik_grasp(obj)
print("kinematic IK: q=", np.round(q,4))

arm_act_idx = np.array([mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j+'_act') for j in env.ARM_JOINTS])
arm_dof = env.arm_dofadr
kp = env.model.actuator_gainprm[arm_act_idx, 0]

env.set_joints(env.q_home)
env.set_gripper(open=True)

def ctrl_with_gravcomp(q_target):
    mujoco.mj_forward(env.model, env.data)
    grav = env.data.qfrc_gravcomp[arm_dof].copy()
    return q_target + grav / kp

for i in range(3000):
    c = ctrl_with_gravcomp(q)
    env.data.ctrl[:] = np.concatenate([c, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)
print("after gravcomp tracking: tooth_mid=", np.round(env.tooth_midpoint(),4), "target=", obj, "err=", np.round(obj-env.tooth_midpoint(),4))
print("  actual q=", np.round(env.get_joints(),4), " target q=", np.round(q,4))

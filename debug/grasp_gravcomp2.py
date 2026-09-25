# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
obj = np.array([1.4, 0.12, 0.36])
q, err = env.solve_ik_grasp(obj)

arm_act_idx = np.array([mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j+'_act') for j in env.ARM_JOINTS])
arm_dof = env.arm_dofadr
kp = env.model.actuator_gainprm[arm_act_idx, 0]

env.set_joints(env.q_home)
env.set_gripper(open=True)

# settle with plain tracking first, measure gravity torque
for _ in range(2000):
    env.data.ctrl[:] = np.concatenate([q, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)
mujoco.mj_forward(env.model, env.data)
print("plain settle: q=", np.round(env.get_joints(),4))
print("  gravcomp=", np.round(env.data.qfrc_gravcomp[arm_dof],2))
print("  qfrc_actuator=", np.round(env.data.qfrc_actuator[arm_dof],2))
print("  qfrc_bias=", np.round(env.data.qfrc_bias[arm_dof],2))
# gravity torque at target q (kinematic)
env.set_joints(q)
mujoco.mj_forward(env.model, env.data)
print("gravcomp at target q=", np.round(env.data.qfrc_gravcomp[arm_dof],2))
print("kp=", kp)
print("required ctrl offset (grav/kp)=", np.round(env.data.qfrc_gravcomp[arm_dof]/kp,4))

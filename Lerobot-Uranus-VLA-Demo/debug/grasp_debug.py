# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
print("after init: grip_qpos=", env.data.qpos[env.model.jnt_qposadr[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,'jaw1.1')]])
ja = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'jaw1.1_act')
jb = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'jaw2.1_act')
print("ctrl jaw1.1_act=", env.data.ctrl[ja], "jaw2.1_act=", env.data.ctrl[jb])

env.set_gripper(open=True)
print("after set_gripper(open): ctrl=", env.data.ctrl[ja], env.data.ctrl[jb])

# step with arm home + open
for i in range(500):
    env.data.ctrl[:] = np.concatenate([env.q_home, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)
print("after 500 open steps: gap=", round(env.tooth_gap()*1000,1), "grip_qpos=", env.data.qpos[env.model.jnt_qposadr[mujoco.mj_name2id(env.model,mujoco.mjtObj.mjOBJ_JOINT,'jaw1.1')]])

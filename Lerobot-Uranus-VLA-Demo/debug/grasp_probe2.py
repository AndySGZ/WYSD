# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)

def show(tag):
    print(f"{tag}: obj={np.round(env.get_object_pos(),4)} tooth_mid={np.round(env.tooth_midpoint(),4)} gap={round(env.tooth_gap()*1000,1)}mm grip={round(env.get_gripper(),4)}")

show("start")
for _ in range(1000):
    env.data.ctrl[:] = np.concatenate([env.q_home, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)
show("after settle 1000 (arm home, open)")

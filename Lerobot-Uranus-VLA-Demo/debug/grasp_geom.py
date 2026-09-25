# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
env.set_joints(env.q_home)
env.set_gripper(open=True)
for _ in range(500):
    env.data.ctrl[:] = np.concatenate([env.q_home, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)

t1 = env.tooth1_id; t2 = env.tooth2_id
p1 = env.data.geom_xpos[t1]; p2 = env.data.geom_xpos[t2]
print("tooth1 center=", np.round(p1,4))
print("tooth2 center=", np.round(p2,4))
print("vector t2-t1=", np.round(p2-p1,4), "gap=", round(np.linalg.norm(p2-p1)*1000,1), "mm")
# orientations
r1 = env.data.geom_xmat[t1].reshape(3,3)
r2 = env.data.geom_xmat[t2].reshape(3,3)
print("tooth1 xmat columns (world axes of geom frame):")
print(np.round(r1,3))
print("tooth2 xmat:")
print(np.round(r2,3))
# object
print("object pos=", np.round(env.get_object_pos(),4))

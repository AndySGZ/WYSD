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
mujoco.mj_forward(env.model, env.data)

t1 = env.tooth1_id; t2 = env.tooth2_id
def gapvec():
    return env.data.geom_xpos[t2] - env.data.geom_xpos[t1]

print("home gap direction:", np.round(gapvec()/np.linalg.norm(gapvec()),3))

# vary each joint and see gap direction
for jname in ['joint4','joint5','joint6']:
    for d in [0.5, 1.0, 1.5, -0.5]:
        q = env.q_home.copy()
        idx = env.ARM_JOINTS.index(jname)
        q[idx] += d
        q = np.clip(q, env.q_min, env.q_max)
        env.set_joints(q)
        g = gapvec()/np.linalg.norm(gapvec())
        print(f"{jname} += {d:+0.1f}: gap_dir={np.round(g,3)}  tm={np.round(env.tooth_midpoint(),3)}")

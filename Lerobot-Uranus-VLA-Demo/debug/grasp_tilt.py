# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
t1=env.tooth1_id
def long_axis():
    return env.data.geom_xmat[t1].reshape(3,3)[:,0]  # local x (long axis)

def gapdir():
    g = env.data.geom_xpos[env.tooth2_id] - env.data.geom_xpos[env.tooth1_id]
    return g/np.linalg.norm(g)

# baseline grasp pose
q, _ = env.solve_ik_grasp(np.array([1.4,0.12,0.36]), q0=env.q_home)
print("baseline q=", np.round(q,3))
print("  long_axis=", np.round(long_axis(),3), " gapdir=", np.round(gapdir(),3))

# vary joint4 and joint5 from baseline, observe long axis z and gapdir
for jname in ['joint4','joint5']:
    idx = env.ARM_JOINTS.index(jname)
    for d in [-0.3, -0.15, 0.0, 0.15, 0.3]:
        qq = q.copy(); qq[idx] += d; qq = np.clip(qq, env.q_min, env.q_max)
        env.set_joints(qq)
        la = long_axis(); gd = gapdir()
        print(f"{jname}{d:+0.2f}: long_axis.z={la[2]:+.3f} gapdir.z={gd[2]:+.3f} tm={np.round(env.tooth_midpoint(),3)}")

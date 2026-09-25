# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)

print("=== reset state ===")
print("obj=", np.round(env.get_object_pos(),3), "tooth_mid=", np.round(env.tooth_midpoint(),3), "gap=", round(env.tooth_gap()*1000,1))

# 1. open gripper dynamics
env.set_gripper(open=True)
for _ in range(200):
    env.data.ctrl[:6] = env.q_home
    mujoco.mj_step(env.model, env.data)
print("after open: gap=", round(env.tooth_gap()*1000,1), "gripper=", round(env.get_gripper(),3))

# 2. IK to approach above object
obj = env.get_object_pos()
above = obj + np.array([0,0,0.08])
q, err = env.solve_ik_grasp(above)
print("IK above: q=", np.round(q,3), "err=", np.round(err,3), "tooth_mid=", np.round(env.tooth_midpoint(),3))

# 3. move dynamics to above
env.move(q, gripper_open=True, substeps=500)
print("after move above: tooth_mid=", np.round(env.tooth_midpoint(),3), "target=", np.round(above,3))

# 4. IK to object
q2, err2 = env.solve_ik_grasp(obj)
print("IK obj: err=", np.round(err2,3), "tooth_mid=", np.round(env.tooth_midpoint(),3))
env.move(q2, gripper_open=True, substeps=500)
print("after move obj: tooth_mid=", np.round(env.tooth_midpoint(),3), "obj=", np.round(env.get_object_pos(),3), "gap=", round(env.tooth_gap()*1000,1))

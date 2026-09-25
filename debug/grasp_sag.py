# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
obj = np.array([1.4, 0.12, 0.36])
q, err = env.solve_ik_grasp(obj)
print("kinematic IK: q=", np.round(q,4), "tooth_mid=", np.round(env.tooth_midpoint(),4))

env.set_joints(env.q_home)
env.set_gripper(open=True)
for i in range(3000):
    env.data.ctrl[:] = np.concatenate([q, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)
tm = env.tooth_midpoint()
print("after tracking: tooth_mid=", np.round(tm,4), "target=", obj, "err=", np.round(obj-tm,4), "|err|mm=", round(np.linalg.norm(obj-tm)*1000,1))
print("  actual q=", np.round(env.get_joints(),4))
print("  obj=", np.round(env.get_object_pos(),4), "gap=", round(env.tooth_gap()*1000,1))

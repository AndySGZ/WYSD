# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)

def settle(q_target, gripper_open, steps=1000):
    env.data.ctrl[:] = np.concatenate([q_target, env.GRIPPER_OPEN if gripper_open else env.GRIPPER_CLOSE])
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)

def show(tag):
    print(f"{tag}: obj={np.round(env.get_object_pos(),3)} tm={np.round(env.tooth_midpoint(),3)} gap={round(env.tooth_gap()*1000,1)} grip={round(env.get_gripper(),3)}")

env.reset()
settle(env.q_home, True, 800)
show("home")

obj = env.get_object_pos()
above = obj + np.array([0,0,0.09])

q, e = env.solve_ik_grasp(above, q0=env.get_joints())
settle(q, True, 1200); show("above open")

q, e = env.solve_ik_grasp(obj + np.array([0,0,0.01]), q0=env.get_joints())
settle(q, True, 1200); show("at obj open")

settle(q, False, 1500); show("closed")

lift = obj + np.array([0,0,0.12])
q, e = env.solve_ik_grasp(lift, q0=env.get_joints())
settle(q, False, 1200); show("lift")

place = env.get_place_target()
p_above = place + np.array([0,0,0.09])
q, e = env.solve_ik_grasp(p_above, q0=env.get_joints())
settle(q, False, 1500); show("place above")

q, e = env.solve_ik_grasp(place + np.array([0,0,0.01]), q0=env.get_joints())
settle(q, False, 1200); show("place at")

settle(q, True, 1000); show("place open")
print("success=", env.success())

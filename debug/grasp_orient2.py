# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
t1=env.tooth1_id; t2=env.tooth2_id
env.set_joints(env.q_home)
env.set_gripper(open=True)
mujoco.mj_forward(env.model, env.data)

def axes(gid):
    m = env.data.geom_xmat[gid].reshape(3,3)
    return m  # columns are geom local axes in world

# home
print("=== home ===")
r1 = axes(t1)
print("tooth1 local x (long) =", np.round(r1[:,0],3))
print("tooth1 local y (thick/face) =", np.round(r1[:,1],3))
print("tooth1 local z (height) =", np.round(r1[:,2],3))

# grasp pose
q, _ = env.solve_ik_grasp(np.array([1.4,0.12,0.36]), q0=env.q_home)
print("=== grasp pose (joint6=pi/2) ===")
r1 = axes(t1)
print("tooth1 local x (long) =", np.round(r1[:,0],3))
print("tooth1 local y (thick/face) =", np.round(r1[:,1],3))
print("tooth1 local z (height) =", np.round(r1[:,2],3))
p1 = env.data.geom_xpos[t1]; p2 = env.data.geom_xpos[t2]
print("gap dir =", np.round((p2-p1)/np.linalg.norm(p2-p1),3))

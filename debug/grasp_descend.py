# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
obj0 = np.array([1.4, 0.12, 0.36])

# settle arm home + open
env.set_joints(env.q_home)
env.set_gripper(open=True)
for _ in range(1000):
    env.data.ctrl[:] = np.concatenate([env.q_home, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)
print("settled home: obj=", np.round(env.get_object_pos(),4), "tm=", np.round(env.tooth_midpoint(),4))

# descend in 5 steps from z=0.44 to 0.36
for z in [0.44, 0.42, 0.40, 0.38, 0.37, 0.36]:
    target = np.array([1.4, 0.12, z])
    q, err = env.solve_ik_grasp(target, q0=env.get_joints())
    for _ in range(500):
        env.data.ctrl[:] = np.concatenate([q, env.GRIPPER_OPEN])
        mujoco.mj_step(env.model, env.data)
    # check contacts on object
    ncon = 0
    for c in env.data.contact:
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        if 'object' in str(g1) or 'object' in str(g2):
            ncon += 1
    print(f"z={z}: obj={np.round(env.get_object_pos(),4)} tm={np.round(env.tooth_midpoint(),4)} gap={round(env.tooth_gap()*1000,1)} obj_contacts={ncon}")

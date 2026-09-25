# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
env.set_joints(env.q_home)
env.set_gripper(open=True)
for _ in range(1000):
    env.data.ctrl[:] = np.concatenate([env.q_home, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)

# descend to z=0.38
target = np.array([1.4, 0.12, 0.38])
q, err = env.solve_ik_grasp(target, q0=env.get_joints())
for _ in range(500):
    env.data.ctrl[:] = np.concatenate([q, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)

obj_id = env.object_body_id
print("obj=", np.round(env.get_object_pos(),4), "tm=", np.round(env.tooth_midpoint(),4), "gap=", round(env.tooth_gap()*1000,1))
print("--- contacts involving object ---")
for c in env.data.contact:
    b1 = env.model.geom_bodyid[c.geom1]
    b2 = env.model.geom_bodyid[c.geom2]
    if b1 == obj_id or b2 == obj_id:
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        body1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b1)
        body2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, b2)
        print(f"  {g1}({body1}) <-> {g2}({body2}) dist={c.dist*1000:.2f}mm")

# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
obj_id = env.object_body_id
t1=env.tooth1_id; t2=env.tooth2_id

def settle(q_target, gripper_open, steps=1000):
    env.data.ctrl[:] = np.concatenate([q_target, env.GRIPPER_OPEN if gripper_open else env.GRIPPER_CLOSE])
    for _ in range(steps):
        mujoco.mj_step(env.model, env.data)

env.reset(); settle(env.q_home, True, 800)
obj = env.get_object_pos()
q, _ = env.solve_ik_grasp(obj + np.array([0,0,0.01]), q0=env.get_joints())
settle(q, True, 1200)
settle(q, False, 1500)

print("after close:")
print("  obj=", np.round(env.get_object_pos(),4))
print("  t1=", np.round(env.data.geom_xpos[t1],4))
print("  t2=", np.round(env.data.geom_xpos[t2],4))
print("  gap=", round(env.tooth_gap()*1000,1), "grip=", round(env.get_gripper(),4))

# contacts between teeth and object
print("  tooth-object contacts:")
for c in env.data.contact:
    b1=env.model.geom_bodyid[c.geom1]; b2=env.model.geom_bodyid[c.geom2]
    if (b1==obj_id and c.geom2 in (t1,t2)) or (b2==obj_id and c.geom1 in (t1,t2)):
        g1 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1)
        g2 = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)
        print(f"    {g1}<->{g2} dist={c.dist*1000:.3f}mm frame={c.frame}")

# contact force on object
mujoco.mj_forward(env.model, env.data)
# net contact force on object via qfrc_constraint for freejoint dofs
print("  obj qfrc_constraint (freejoint 0:6)=", np.round(env.data.qfrc_constraint[:6],3))
print("  obj qvel (0:6)=", np.round(env.data.qvel[:6],4))

# try to lift: command arm up but keep gripper closed
lift = obj + np.array([0,0,0.12])
q2, _ = env.solve_ik_grasp(lift, q0=env.get_joints())
for i in range(1000):
    env.data.ctrl[:] = np.concatenate([q2, env.GRIPPER_CLOSE])
    mujoco.mj_step(env.model, env.data)
    if i % 250 == 0:
        print(f"  lift i={i}: obj={np.round(env.get_object_pos(),4)} gap={round(env.tooth_gap()*1000,1)}")

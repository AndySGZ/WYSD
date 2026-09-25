# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
t1=env.tooth1_id; t2=env.tooth2_id
def solve5(target, q0):
    q = q0.copy(); q[5]=np.pi/2
    for _ in range(400):
        env.set_joints(q)
        e = target - env.tooth_midpoint()
        if np.linalg.norm(e)<1e-3: break
        jp1=np.zeros((3,env.model.nv)); jp2=np.zeros((3,env.model.nv)); jp=np.zeros((3,env.model.nv))
        mujoco.mj_jac(env.model,env.data,jp1,None,env.data.geom_xpos[t1],env.jaw1_body_id)
        mujoco.mj_jac(env.model,env.data,jp2,None,env.data.geom_xpos[t2],env.jaw2_body_id)
        jp[:]=0.5*(jp1+jp2)
        J5=jp[:,env.arm_dofadr[:5]]
        dq=np.clip(np.linalg.lstsq(J5,e,rcond=1e-3)[0],-0.3,0.3)
        q[:5]=np.clip(q[:5]+dq,env.q_min[:5],env.q_max[:5])
    env.set_joints(q)
    return q

obj_id = env.object_body_id
def obj_contacts():
    names=[]
    for c in env.data.contact:
        b1=env.model.geom_bodyid[c.geom1]; b2=env.model.geom_bodyid[c.geom2]
        if b1==obj_id or b2==obj_id:
            g = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1 if b1!=obj_id else c.geom2)
            names.append(g)
    return names

qg = solve5(np.array([1.4,0.12,0.37]), env.q_home)
env.set_joints(qg)
env.set_gripper(open=True)
for _ in range(800):
    env.data.ctrl[:] = np.concatenate([qg, env.GRIPPER_OPEN])
    mujoco.mj_step(env.model, env.data)
print("open settled: gap=", round(env.tooth_gap()*1000,1), "grip=", round(env.get_gripper(),3), "tm=", np.round(env.tooth_midpoint(),3), "obj=", np.round(env.get_object_pos(),3), "contacts=", obj_contacts())

# close with monitoring
for i in range(2000):
    env.data.ctrl[:] = np.concatenate([qg, env.GRIPPER_CLOSE])
    mujoco.mj_step(env.model, env.data)
    if i % 400 == 0:
        print(f"close i={i}: gap={round(env.tooth_gap()*1000,1)} grip={round(env.get_gripper(),4)} t1={np.round(env.data.geom_xpos[t1],3)} t2={np.round(env.data.geom_xpos[t2],3)} contacts={obj_contacts()}")

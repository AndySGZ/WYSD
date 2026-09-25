# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)

def run_test(tag, q_arm):
    env.set_joints(q_arm)
    env.set_gripper(open=True)
    for _ in range(500):
        env.data.ctrl[:] = np.concatenate([q_arm, env.GRIPPER_OPEN])
        mujoco.mj_step(env.model, env.data)
    g_open = env.tooth_gap()
    # now close
    for i in range(1500):
        env.data.ctrl[:] = np.concatenate([q_arm, env.GRIPPER_CLOSE])
        mujoco.mj_step(env.model, env.data)
        if i % 300 == 0:
            print(f"  {tag} close step {i}: gap={round(env.tooth_gap()*1000,1)} grip={round(env.get_gripper(),4)}")
    print(f"{tag}: open gap={round(g_open*1000,1)} -> final gap={round(env.tooth_gap()*1000,1)}")

print("=== home (vertical opening) ===")
run_test("home", env.q_home)

# horizontal opening pose
q_h, _ = env.solve_ik_grasp(np.array([1.4,0.12,0.44]))
q_h = q_h.copy(); q_h[5] = np.pi/2
# re-solve 5dof quickly
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
qg = solve5(np.array([1.4,0.12,0.44]), env.q_home)
print("=== horizontal opening pose ===")
run_test("horiz", qg)

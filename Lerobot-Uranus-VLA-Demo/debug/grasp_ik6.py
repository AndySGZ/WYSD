# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
t1=env.tooth1_id; t2=env.tooth2_id
tcp_id = env.tcp_body_id

def rotvec(R):
    c = np.clip((np.trace(R)-1)/2, -1, 1)
    th = np.arccos(c)
    if th < 1e-10:
        return np.zeros(3)
    v = np.array([R[2,1]-R[1,2], R[0,2]-R[2,0], R[1,0]-R[0,1]])/(2*np.sin(th))
    return th*v

def solve_ik6(target, q0, max_iter=500, tol=1e-3, rot_gain=1.0):
    q = np.asarray(q0, dtype=float).copy()
    R_des = np.eye(3)
    jacp1 = np.zeros((3, env.model.nv)); jacp2 = np.zeros((3, env.model.nv))
    jacp_b = np.zeros((3, env.model.nv)); jacr = np.zeros((3, env.model.nv))
    for _ in range(max_iter):
        env.set_joints(q)
        tm = env.tooth_midpoint()
        epos = target - tm
        R_cur = env.data.geom_xmat[t1].reshape(3,3)
        eori = rotvec(R_des @ R_cur.T) * rot_gain
        err = np.concatenate([epos, eori])
        if np.linalg.norm(epos) < tol and np.linalg.norm(eori) < 0.01:
            break
        mujoco.mj_jac(env.model, env.data, jacp1, None, env.data.geom_xpos[t1], env.jaw1_body_id)
        mujoco.mj_jac(env.model, env.data, jacp2, None, env.data.geom_xpos[t2], env.jaw2_body_id)
        mujoco.mj_jacBody(env.model, env.data, jacp_b, jacr, tcp_id)
        J = np.zeros((6, 6))
        J[:3] = 0.5*(jacp1+jacp2)[:, env.arm_dofadr]
        J[3:] = jacr[:, env.arm_dofadr]
        dq = np.linalg.lstsq(J, err, rcond=1e-3)[0]
        dq = np.clip(dq, -0.2, 0.2)
        q = np.clip(q + dq, env.q_min, env.q_max)
    env.set_joints(q)
    R_cur = env.data.geom_xmat[t1].reshape(3,3)
    return q, target - env.tooth_midpoint(), R_cur[:,0]

q, epos, la = solve_ik6(np.array([1.4,0.12,0.36]), env.q_home)
print("6dof IK: q=", np.round(q,4))
print("  epos=", np.round(epos,4), "long_axis=", np.round(la,3))
print("  tm=", np.round(env.tooth_midpoint(),3))
gd = env.data.geom_xpos[t2]-env.data.geom_xpos[t1]; gd/=np.linalg.norm(gd)
print("  gapdir=", np.round(gd,3), "gap=", round(env.tooth_gap()*1000,1))

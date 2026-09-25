# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r'E:\projects\mujoco\Lerobot-Uranus-VLA-Demo')
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
t1 = env.tooth1_id; t2 = env.tooth2_id

def solve_ik_5dof(target, q0, j6, max_iter=400, tol=1e-3):
    q = np.asarray(q0, dtype=float).copy()
    q[5] = j6
    jacp = np.zeros((3, env.model.nv))
    for _ in range(max_iter):
        env.set_joints(q)
        err = target - env.tooth_midpoint()
        if np.linalg.norm(err) < tol:
            break
        jacp1 = np.zeros((3, env.model.nv)); jacp2 = np.zeros((3, env.model.nv))
        mujoco.mj_jac(env.model, env.data, jacp1, None, env.data.geom_xpos[t1], env.jaw1_body_id)
        mujoco.mj_jac(env.model, env.data, jacp2, None, env.data.geom_xpos[t2], env.jaw2_body_id)
        jacp[:] = 0.5*(jacp1+jacp2)
        J5 = jacp[:, env.arm_dofadr[:5]]  # only joints 1..5
        dq = np.linalg.lstsq(J5, err, rcond=1e-3)[0]
        dq = np.clip(dq, -0.3, 0.3)
        q[:5] = np.clip(q[:5] + dq, env.q_min[:5], env.q_max[:5])
    env.set_joints(q)
    return q, target - env.tooth_midpoint()

def gapdir():
    g = env.data.geom_xpos[t2] - env.data.geom_xpos[t1]
    return g/np.linalg.norm(g)

target = np.array([1.4, 0.12, 0.44])
q, err = solve_ik_5dof(target, env.q_home, j6=np.pi/2)
print("5dof IK: q=", np.round(q,3), "err=", np.round(err,4))
print("  tm=", np.round(env.tooth_midpoint(),3), "gapdir=", np.round(gapdir(),3), "gap=", round(env.tooth_gap()*1000,1))

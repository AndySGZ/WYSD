# -*- coding: utf-8 -*-
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import mujoco
from src.env_grasp import UranusGraspEnv

t1=None; t2=None
def solve5(env, target, q0):
    q = q0.copy(); q[5]=np.pi/2
    for _ in range(400):
        env.set_joints(q)
        e = target - env.tooth_midpoint()
        if np.linalg.norm(e)<1e-3: break
        jp1=np.zeros((3,env.model.nv)); jp2=np.zeros((3,env.model.nv)); jp=np.zeros((3,env.model.nv))
        mujoco.mj_jac(env.model,env.data,jp1,None,env.data.geom_xpos[env.tooth1_id],env.jaw1_body_id)
        mujoco.mj_jac(env.model,env.data,jp2,None,env.data.geom_xpos[env.tooth2_id],env.jaw2_body_id)
        jp[:]=0.5*(jp1+jp2)
        J5=jp[:,env.arm_dofadr[:5]]
        dq=np.clip(np.linalg.lstsq(J5,e,rcond=1e-3)[0],-0.3,0.3)
        q[:5]=np.clip(q[:5]+dq,env.q_min[:5],env.q_max[:5])
    env.set_joints(q)
    return q

def run(jaw_kp):
    env = UranusGraspEnv(seed=0)
    jaw_act = [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, j+'_act') for j in env.GRIPPER_JOINTS]
    env.model.actuator_gainprm[jaw_act,0] = jaw_kp
    env.model.actuator_biasprm[jaw_act,1] = -jaw_kp
    env.model.actuator_biasprm[jaw_act,2] = -jaw_kp*0.05
    qg = solve5(env, np.array([1.4,0.12,0.37]), env.q_home)
    env.set_joints(qg); env.set_gripper(open=True)
    for _ in range(800):
        env.data.ctrl[:] = np.concatenate([qg, env.GRIPPER_OPEN]); mujoco.mj_step(env.model, env.data)
    open_gap = env.tooth_gap()
    # close
    for _ in range(1500):
        env.data.ctrl[:] = np.concatenate([qg, env.GRIPPER_CLOSE]); mujoco.mj_step(env.model, env.data)
    obj = env.get_object_pos()
    grip = env.get_gripper()
    gap = env.tooth_gap()
    # is object lifted/gripped? check if object moved from rest
    print(f"jaw_kp={jaw_kp:6d}: open_gap={round(open_gap*1000,1)} close_gap={round(gap*1000,1)} grip={round(grip,4)} obj={np.round(obj,3)}")

for kp in [200, 500, 1000, 2000, 5000]:
    run(kp)

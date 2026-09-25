import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import mujoco
import numpy as np

ROOT = Path.cwd()
m = mujoco.MjModel.from_xml_path(str(ROOT / "asset/scene_uranus_grasp.xml"))
d = mujoco.MjData(m)
arm = ["joint1","joint2","joint3","joint4","joint5","joint6"]
arm_qpos = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm])
arm_dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm])
qmin = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j),0] for j in arm])
qmax = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j),1] for j in arm])
t1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "jaw1_tooth_col")
t2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "jaw2_tooth_col")
jb1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw1.1.1")
jb2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw2.1.1")
home_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")

def setj(q): d.qpos[arm_qpos] = np.clip(q, qmin, qmax); mujoco.mj_forward(m, d)
def mid(): return (d.geom_xpos[t1] + d.geom_xpos[t2]) / 2
def ik(target, q0=None):
    q = (d.qpos[arm_qpos].copy() if q0 is None else q0.copy())
    for _ in range(400):
        setj(q); err = target - mid()
        if np.linalg.norm(err) < 1e-3: break
        j1 = np.zeros((3, m.nv)); j2 = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, j1, None, d.geom_xpos[t1], jb1)
        mujoco.mj_jac(m, d, j2, None, d.geom_xpos[t2], jb2)
        J6 = 0.5*(j1+j2)[:, arm_dof]
        dq = np.clip(np.linalg.lstsq(J6, err, rcond=1e-3)[0], -0.3, 0.3)
        q = np.clip(q + dq, qmin, qmax)
    setj(q); return q

def move(q, open_, n=300):
    d.ctrl[:6] = q; d.ctrl[6] = 0.3 if open_ else -0.2
    for _ in range(n): mujoco.mj_step(m, d)

mujoco.mj_resetDataKeyframe(m, d, home_key); mujoco.mj_forward(m, d)
target = np.array([1.4, 0.12, 0.36])
q = ik(target)
print("initial ik mid err =", round(np.linalg.norm(mid()-target)*1000,2), "mm")

# closed-loop correction
for it in range(5):
    move(q, True, 300)
    err = target - mid()
    e = np.linalg.norm(err)
    print(f"  iter {it}: mid err = {e*1000:5.1f} mm, mid={np.round(mid(),3).tolist()}")
    if e < 0.002:
        break
    target2 = target + err  # over-command
    q = ik(target2, q)
    print(f"          re-IK to {np.round(target2,3).tolist()}, kin err={np.linalg.norm(mid()-target2)*1000:.2f}mm")

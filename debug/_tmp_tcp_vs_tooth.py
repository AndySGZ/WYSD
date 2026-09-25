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
tcp = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "tcp_link")
home_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
jaw = ["jaw1.1","jaw1.1.1","jaw1.2","jaw2.2","jaw2.1","jaw2.1.1"]
jaw_qpos = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in jaw])

def setj(q): d.qpos[arm_qpos] = np.clip(q, qmin, qmax); mujoco.mj_forward(m, d)
def mid(): return (d.geom_xpos[t1] + d.geom_xpos[t2]) / 2
def ik(target):
    q = d.qpos[arm_qpos].copy()
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

mujoco.mj_resetDataKeyframe(m, d, home_key); mujoco.mj_forward(m, d)
target = np.array([1.4, 0.12, 0.36])
q = ik(target)
print("KINEMATIC: tcp =", np.round(d.xpos[tcp],3).tolist(), "tooth_mid =", np.round(mid(),3).tolist())
print("           jaws =", np.round(d.qpos[jaw_qpos],3).tolist(), "tendon =", round(float(d.sensor("gripper_position").data),3))

# settle with gripper OPEN ctrl 0.3
d.ctrl[:6] = q; d.ctrl[6] = 0.3
for _ in range(2500): mujoco.mj_step(m, d)
print("SETTLE(open): tcp =", np.round(d.xpos[tcp],3).tolist(), "tooth_mid =", np.round(mid(),3).tolist())
print("              jaws =", np.round(d.qpos[jaw_qpos],3).tolist(), "tendon =", round(float(d.sensor("gripper_position").data),3))
print("              tooth gap =", round(np.linalg.norm(d.geom_xpos[t1]-d.geom_xpos[t2])*1000,1), "mm")

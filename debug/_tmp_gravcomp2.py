import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import mujoco
import numpy as np

ROOT = Path.cwd()
m = mujoco.MjModel.from_xml_path(str(ROOT / "asset/scene_uranus_grasp.xml"))
d = mujoco.MjData(m)
arm = ["joint1","joint2","joint3","joint4","joint5","joint6"]
act = ["joint1_act","joint2_act","joint3_act","joint4_act","joint5_act","joint6_act"]
arm_qpos = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm])
arm_dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in arm])
act_id = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in act])
qmin = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j),0] for j in arm])
qmax = np.array([m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j),1] for j in arm])
t1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "jaw1_tooth_col")
t2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "jaw2_tooth_col")
jb1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw1.1.1")
jb2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "jaw2.1.1")
home_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, home_key); mujoco.mj_forward(m, d)
q_home = d.qpos[arm_qpos].copy()
def setj(q): d.qpos[arm_qpos] = np.clip(q, qmin, qmax); mujoco.mj_forward(m, d)
def mid(): return (d.geom_xpos[t1] + d.geom_xpos[t2]) / 2
target = np.array([1.4, 0.12, 0.36]); q = q_home.copy()
for _ in range(400):
    setj(q); err = target - mid()
    if np.linalg.norm(err) < 1e-3: break
    j1 = np.zeros((3, m.nv)); j2 = np.zeros((3, m.nv))
    mujoco.mj_jac(m, d, j1, None, d.geom_xpos[t1], jb1)
    mujoco.mj_jac(m, d, j2, None, d.geom_xpos[t2], jb2)
    J6 = 0.5*(j1+j2)[:, arm_dof]
    dq = np.clip(np.linalg.lstsq(J6, err, rcond=1e-3)[0], -0.3, 0.3)
    q = np.clip(q + dq, qmin, qmax)
setj(q)
kp = m.actuator_gainprm[act_id,0]
g_bias = d.qfrc_bias[arm_dof].copy()
g_grav = d.qfrc_gravcomp[arm_dof].copy()
print("qfrc_bias     =", np.round(g_bias,2).tolist())
print("qfrc_gravcomp =", np.round(g_grav,2).tolist())
print("kp =", kp)

def settle(ctrl_arm, n=2500):
    setj(q)
    d.ctrl[:6] = ctrl_arm
    d.ctrl[6] = 0.3
    for _ in range(n): mujoco.mj_step(m, d)
    return np.linalg.norm(mid()-target)*1000, d.qpos[arm_qpos].copy()

for name, ctrl in [
    ("no comp", q),
    ("+bias/kp", q + g_bias/kp),
    ("-bias/kp", q - g_bias/kp),
    ("+grav/kp", q + g_grav/kp),
    ("-grav/kp", q - g_grav/kp),
]:
    err, qa = settle(ctrl)
    print(f"{name:12s}: mid err = {err:5.1f} mm   q={np.round(qa,3).tolist()}")

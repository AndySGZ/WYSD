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
t1 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "jaw1_tooth_col")
t2 = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "jaw2_tooth_col")
home_key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
jaw = ["jaw1.1","jaw1.2","jaw2.1","jaw2.2"]
jaw_qpos = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in jaw])

def test(q_arm, label):
    mujoco.mj_resetDataKeyframe(m, d, home_key); mujoco.mj_forward(m, d)
    d.qpos[arm_qpos] = q_arm
    mujoco.mj_forward(m, d)
    d.ctrl[:6] = q_arm
    d.ctrl[6] = 0.3  # open
    for _ in range(2000): mujoco.mj_step(m, d)
    gap = np.linalg.norm(d.geom_xpos[t1]-d.geom_xpos[t2])
    print(f"{label:22s}: gap={gap*1000:5.1f}mm tendon={float(d.sensor('gripper_position').data):6.3f} jaws={np.round(d.qpos[jaw_qpos],2).tolist()}")

test(np.array([0.0, -0.8122, -1.4859, 0.7267, 0.0, 0.0]), "home (upright)")
test(np.array([0.08, -0.5953, -1.9617, 0.701, 0.0165, 0.0]), "grasp pose")
test(np.array([0.0, -0.7, -1.6, 0.7, 0.0, 0.0]), "intermediate")

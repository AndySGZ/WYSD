import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mujoco
import numpy as np

ROOT = Path.cwd()
ARM = ROOT / "asset/uranus/uranus_arm_gripper_model.xml"
SCENE = ROOT / "asset/scene_uranus_grasp.xml"
orig = ARM.read_text(encoding="utf-8")

def set_gripper(txt, kp, kv, fr):
    txt = re.sub(r'(<position name="gripper_act"[^>]*kp=")\d+(" kv=")\d+(")', rf'\g<1>{kp}\g<2>{kv}\g<3>', txt)
    if fr:
        txt = re.sub(r'forcerange="[^"]*"', f'forcerange="{fr}"', txt)
    return txt

def measure():
    m = mujoco.MjModel.from_xml_path(str(SCENE)); d = mujoco.MjData(m)
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
    jaw = ["jaw1.1","jaw1.1.1","jaw1.2","jaw2.2","jaw2.1","jaw2.1.1"]
    jaw_qpos = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in jaw])
    def setj(q): d.qpos[arm_qpos] = np.clip(q, qmin, qmax); mujoco.mj_forward(m, d)
    def mid(): return (d.geom_xpos[t1] + d.geom_xpos[t2]) / 2
    mujoco.mj_resetDataKeyframe(m, d, home_key); mujoco.mj_forward(m, d)
    target = np.array([1.4, 0.12, 0.36]); q = d.qpos[arm_qpos].copy()
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
    d.ctrl[:6] = q; d.ctrl[6] = 0.3
    for _ in range(2500): mujoco.mj_step(m, d)
    gap = np.linalg.norm(d.geom_xpos[t1]-d.geom_xpos[t2])
    return np.linalg.norm(mid()-target)*1000, gap*1000, float(d.sensor("gripper_position").data), d.qpos[jaw_qpos].copy()

try:
    configs = [
        ("kp20 fr2 (baseline)", 20, 5, "-2 2"),
        ("kp50 fr10", 50, 10, "-10 10"),
        ("kp100 fr20", 100, 20, "-20 20"),
        ("kp100 noforcelimit", 100, 20, None),
        ("kp200 noforcelimit", 200, 30, None),
    ]
    for name, kp, kv, fr in configs:
        txt = set_gripper(orig, kp, kv, fr)
        if fr is None:
            txt = txt.replace('forcelimited="true"', 'forcelimited="false"')
        ARM.write_text(txt, encoding="utf-8")
        try:
            err, gap, sens, jw = measure()
            print(f"{name:22s}: mid_err={err:5.1f}mm gap={gap:5.1f}mm tendon={sens:6.3f} jaws={np.round(jw,2).tolist()}")
        except Exception as e:
            print(f"{name:22s}: FAILED {e}")
finally:
    ARM.write_text(orig, encoding="utf-8")
    print("(restored)")

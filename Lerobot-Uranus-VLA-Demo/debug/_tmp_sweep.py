import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mujoco
import numpy as np

ROOT = Path.cwd()
ARM_MODEL = ROOT / "asset/uranus/uranus_arm_gripper_model.xml"
SCENE = ROOT / "asset/scene_uranus_grasp.xml"
orig = ARM_MODEL.read_text(encoding="utf-8")

BASE = {
    "joint1_act": (15000, 500), "joint2_act": (15000, 500), "joint3_act": (15000, 500),
    "joint4_act": (10000, 200), "joint5_act": (8000, 150), "joint6_act": (3000, 100),
}
def set_kpkv(txt, name, kp, kv):
    return re.sub(rf'(<position name="{name}"[^>]*kp=")\d+(" kv=")\d+(")',
                  rf'\g<1>{kp}\g<2>{kv}\g<3>', txt)

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
    setj(q); d.ctrl[:] = np.concatenate([q, [0.3]])
    for _ in range(1500): mujoco.mj_step(m, d)
    return np.linalg.norm(mid() - target), mid()

try:
    configs = [
        ("baseline", None),
        ("all x2", 2.0),
        ("all x3", 3.0),
        ("all x4", 4.0),
        ("all x5", 5.0),
    ]
    for name, f in configs:
        txt = orig
        if f:
            for jn, (kp0, kv0) in BASE.items():
                txt = set_kpkv(txt, jn, int(kp0*f), int(kv0*f))
        ARM_MODEL.write_text(txt, encoding="utf-8")
        try:
            err, mid = measure()
            print(f"{name:12s}: sag = {err*1000:6.1f} mm  mid={np.round(mid,3).tolist()}")
        except Exception as e:
            print(f"{name:12s}: FAILED {e}")
finally:
    ARM_MODEL.write_text(orig, encoding="utf-8")
    print("(restored)")

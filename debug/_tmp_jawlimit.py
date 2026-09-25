import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import mujoco
import numpy as np

ROOT = Path.cwd()
m = mujoco.MjModel.from_xml_path(str(ROOT / "asset/scene_uranus_grasp.xml"))
d = mujoco.MjData(m)
jaw = ["jaw1.1","jaw1.1.1","jaw1.2","jaw2.2","jaw2.1","jaw2.1.1"]
for j in jaw:
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)
    print(f"{j:12s}: range={m.jnt_range[jid]} limited={m.jnt_limited[jid]} ref={m.qpos0[m.jnt_qposadr[jid]]:.3f}")

# check tendon coefs
tend_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TENDON, "jaw")
print("tendon 'jaw' nv-limited etc")
for k in range(m.ntendon):
    print("  tendon", k, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, k))
# tendon wrap? fixed?
print("tendon fixed:", m.tendon_fixed if hasattr(m, 'tendon_fixed') else "N/A")

# actuator
aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper_act")
print("gripper_act: gainprm=", m.actuator_gainprm[aid], "biasprm=", m.actuator_biasprm[aid], "forcerange=", m.actuator_forcerange[aid], "forcelimited=", m.actuator_forcelimited[aid], "ctrlrange=", m.actuator_ctrlrange[aid])
print("actuator trntype:", m.actuator_trntype[aid], "trnid:", m.actuator_trnid[aid])

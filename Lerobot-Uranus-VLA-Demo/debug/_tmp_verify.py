import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mujoco
import numpy as np

for scene in ["asset/scene_uranus.xml", "asset/scene_uranus_grasp.xml"]:
    m = mujoco.MjModel.from_xml_path(str(Path.cwd() / scene))
    d = mujoco.MjData(m)
    print(f"{scene}: nq={m.nq}, nu={m.nu}")
    # joint order
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
    print("  joints:", names)
    home = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, home)
    mujoco.mj_forward(m, d)
    print("  qpos0 after home reset (first 12):", np.round(d.qpos[:12], 4).tolist())
    print()

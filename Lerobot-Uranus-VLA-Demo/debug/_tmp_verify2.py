import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import mujoco
import numpy as np

for scene in ["asset/scene_uranus.xml", "asset/scene_uranus_grasp.xml"]:
    m = mujoco.MjModel.from_xml_path(str(Path.cwd() / scene))
    d = mujoco.MjData(m)
    print(f"{scene}: nq={m.nq}, nu={m.nu}")
    print("  actuators:", [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)])

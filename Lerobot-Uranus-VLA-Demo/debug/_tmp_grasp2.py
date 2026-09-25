import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
obs = env.reset()
obj = env.get_object_pos()
print("obj =", np.round(obj,3).tolist(), "gripper =", round(env.get_gripper(),3), "tooth gap =", round(env.tooth_gap()*1000,1),"mm")

# move to grasp pose (open gripper)
q, err = env.solve_ik_grasp(obj)
print("IK err =", round(np.linalg.norm(err),4))
env.move(q, gripper_open=True, substeps=1500)
print("[open settle] tooth_mid =", np.round(env.tooth_midpoint(),3).tolist(), "target =", np.round(obj,3).tolist())
print("  tooth_mid err =", round(np.linalg.norm(env.tooth_midpoint()-obj)*1000,1), "mm")
print("  tooth gap =", round(env.tooth_gap()*1000,1), "mm  gripper =", round(env.get_gripper(),3))
print("  obj pos =", np.round(env.get_object_pos(),3).tolist())

# close gripper
env.move(q, gripper_open=False, substeps=1000)
print("[close] tooth gap =", round(env.tooth_gap()*1000,1), "mm  gripper =", round(env.get_gripper(),3))
print("  obj pos =", np.round(env.get_object_pos(),3).tolist())

# lift
q_lift, _ = env.solve_ik_grasp(obj + np.array([0,0,0.15]))
env.move(q_lift, gripper_open=False, substeps=1500)
print("[lift] obj pos =", np.round(env.get_object_pos(),3).tolist(), "tooth_mid =", np.round(env.tooth_midpoint(),3).tolist())

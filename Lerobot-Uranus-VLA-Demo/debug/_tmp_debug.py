import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from src.env_grasp import UranusGraspEnv

env = UranusGraspEnv(seed=0)
env.reset()
obj = env.get_object_pos()
q, err = env.solve_ik_grasp(obj)
print("IK err =", np.linalg.norm(err), "q =", np.round(q,4).tolist())
print("tooth_mid after IK =", np.round(env.tooth_midpoint(),3).tolist())

# now dynamics via env.move
env.move(q, gripper_open=True, substeps=1500)
print("after move tooth_mid =", np.round(env.tooth_midpoint(),3).tolist())
print("joints after move =", np.round(env.get_joints(),4).tolist())
print("tcp after move =", np.round(env.get_tcp_pos(),3).tolist())
print("tooth gap =", round(env.tooth_gap()*1000,1), "mm")

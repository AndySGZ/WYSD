import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import mujoco

m = mujoco.MjModel.from_xml_path(str(Path.cwd() / "asset/scene_uranus_grasp.xml"))
print("timestep =", m.opt.timestep)
print("integrator =", m.opt.integrator, "(0=Euler,1=RK4,2=implicit,3=implicitfast)")
print("solver =", m.opt.solver, "(0=PGS,1=CG,2=Newton)")
print("cone =", m.opt.cone)
print("impratio =", m.opt.impratio)
print("iterations =", m.opt.iterations)
print("tolerance =", m.opt.tolerance)
print("gravity =", m.opt.gravity)

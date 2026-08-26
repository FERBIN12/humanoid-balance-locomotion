#!/usr/bin/env python3
"""Print the state BY NAME before trusting it. The habit that prevents
most index bugs: a wrong index gives a controller that nearly works."""
import os, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("%-26s %5s %5s %9s" % ("joint", "qpos", "qvel", "angle"))
for i in range(1, m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    qa, va = m.jnt_qposadr[i], m.jnt_dofadr[i]
    print("%-26s %5d %5d %9.4f" % (n, qa, va, d.qpos[qa]))

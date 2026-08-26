#!/usr/bin/env python3
"""Step the physics and watch time advance. The smallest useful MuJoCo loop."""
import os, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
print("before: t = %.3f s   pelvis z = %.3f m" % (d.time, d.qpos[2]))
for i in range(500):
    mujoco.mj_step(m, d)
print("after 500 steps: t = %.3f s   pelvis z = %.3f m" % (d.time, d.qpos[2]))
print("500 steps x 0.002 s = 1.000 s of simulated time")

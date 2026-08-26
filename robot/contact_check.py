#!/usr/bin/env python3
"""Do not ASSUME the foot is on the ground. Read the contacts."""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
for _ in range(400):
    mujoco.mj_step(m, d)
print("after %.2f s of settling: %d contacts" % (d.time, d.ncon))
tot = 0.0
for i in range(min(d.ncon, 8)):
    f = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f)
    tot += f[0]
    print("  contact %d  normal force %8.1f N" % (i, f[0]))
print("total normal force %.1f N; body weight is %.1f N"
      % (tot, m.body_mass.sum() * 9.81))

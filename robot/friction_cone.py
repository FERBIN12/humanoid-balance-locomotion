#!/usr/bin/env python3
"""The friction cone, and how close each contact is to sliding."""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
for _ in range(600):
    mujoco.mj_step(m, d)

print("%-6s %10s %10s %8s %8s" % ("i", "normal N", "tangent N", "ratio", "margin"))
worst = 0.0
for i in range(min(d.ncon, 10)):
    f = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f)
    mu = d.contact[i].friction[0]
    fn, ft = f[0], float(np.hypot(f[1], f[2]))
    ratio = ft / fn if fn > 1e-6 else 0.0
    worst = max(worst, ratio / mu if mu else 0)
    print("%-6d %10.1f %10.1f %8.3f %8s"
          % (i, fn, ft, ratio, "SLIDING" if ratio > mu else "held"))
mu = d.contact[0].friction[0] if d.ncon else 1.0
print()
print("mu = %.2f, so the cone half angle is %.1f deg"
      % (mu, np.degrees(np.arctan(mu))))
print("worst contact is at %.0f%% of its friction limit" % (100 * worst))
print()
print("this single number decides whether the robot walks or slides on a slope")

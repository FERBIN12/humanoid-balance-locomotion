#!/usr/bin/env python3
"""The three-number contact check I run whenever contact behaves strangely.

  1 how many contacts    (zero means the foot is floating)
  2 total normal force   (compare to weight; absurd means sunk through)
  3 the centre of pressure (outside the foot means you summed the wrong thing)
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
for _ in range(600):
    mujoco.mj_step(m, d)

tot, moment = 0.0, np.zeros(3)
for i in range(d.ncon):
    f = np.zeros(6)
    mujoco.mj_contactForce(m, d, i, f)
    tot += f[0]
    moment += d.contact[i].pos * f[0]

weight = m.body_mass.sum() * 9.81
print("1) contacts            %d" % d.ncon)
print("2) total normal force  %.1f N" % tot)
print("   body weight         %.1f N" % weight)
print("   agreement           %.2f%% off" % (100 * abs(weight - tot) / weight))
cop = moment / tot if tot > 0 else np.zeros(3)
print("3) centre of pressure  %s" % np.round(cop, 4))
print()
print("the CoP is a force weighted average of contact POSITIONS, so it is")
print("inside the convex hull of the contacts BY CONSTRUCTION. It cannot leave.")

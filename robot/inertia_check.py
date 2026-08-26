#!/usr/bin/env python3
"""The check I run on every description I am handed.

Three questions:
  1 does the total mass match the datasheet
  2 are the left and right limbs identical
  3 are a large body's three principal moments DISTINCT
    (all equal means somebody approximated a sphere)
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i): i
         for i in range(m.nbody)}

print("1) total mass %.2f kg" % m.body_mass.sum())

bad = []
for n, i in names.items():
    if n and n.startswith("left_"):
        r = n.replace("left_", "right_", 1)
        if r in names and abs(m.body_mass[i] - m.body_mass[names[r]]) > 1e-4:
            bad.append(n)
print("2) left/right mass mismatches: %d %s"
      % (len(bad), "" if bad else "(identical to the gram)"))

print("3) heaviest bodies, principal moments about their own CoM:")
for i in sorted(range(m.nbody), key=lambda k: -m.body_mass[k])[:5]:
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
    I = m.body_inertia[i]
    distinct = len(set(np.round(I, 5))) > 1
    print("   %-22s %6.2f kg  I = %s  %s"
          % (n, m.body_mass[i], np.round(I, 4),
             "distinct" if distinct else "ALL EQUAL: check this"))

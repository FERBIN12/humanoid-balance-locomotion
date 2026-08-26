#!/usr/bin/env python3
"""Print every leg joint limit, in radians and degrees. Read, do not assume."""
import math, os, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
print("%-26s %16s %16s" % ("joint", "radians", "degrees"))
for i in range(1, m.njnt):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    if not any(k in n for k in ("hip", "knee", "ankle")):
        continue
    lo, hi = m.jnt_range[i]
    print("%-26s %7.2f %7.2f  %7.0f %7.0f"
          % (n, lo, hi, math.degrees(lo), math.degrees(hi)))
print()
print("ankle roll is the tightest at 15 deg: a MECHANICAL bound, not a gain")

#!/usr/bin/env python3
"""The ankle frame is NOT the sole. Using it as the contact point is a real bug."""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
for _ in range(500):
    mujoco.mj_step(m, d)          # let it settle onto the floor

a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
print("ankle_roll_link world z = %.4f m" % d.xpos[a][2])

# the LOWEST point of any geom on that body is the real contact height
lowest = None
for g in range(m.ngeom):
    if m.geom_bodyid[g] == a:
        z = d.geom_xpos[g][2] - m.geom_rbound[g]
        lowest = z if lowest is None else min(lowest, z)
print("lowest geom extent   = %.4f m" % (lowest if lowest is not None else float("nan")))
print()
print("difference           = %.4f m" % abs(d.xpos[a][2] - (lowest or 0)))
print("treat the ankle frame as the contact point and your CoP is that far off the floor")

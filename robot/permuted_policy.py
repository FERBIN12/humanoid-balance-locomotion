#!/usr/bin/env python3
"""Sorting joint names alphabetically feels tidy and breaks a trained policy."""
import os, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
model_order = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
               for i in range(m.nu)]
alpha = sorted(model_order)
print("%-3s %-26s %-26s" % ("i", "MODEL ORDER", "ALPHABETICAL"))
for i, (a, b) in enumerate(zip(model_order, alpha)):
    flag = "" if a == b else "  <-- MISMATCH"
    print("%-3d %-26s %-26s%s" % (i, a, b, flag))
n = sum(1 for a, b in zip(model_order, alpha) if a != b)
print()
print("%d of %d positions differ" % (n, m.nu))
print("a policy trained on model order, fed alphabetical order, still outputs")
print("twelve plausible numbers and puts the robot on the floor")

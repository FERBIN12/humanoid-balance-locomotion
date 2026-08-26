#!/usr/bin/env python3
"""The bug that looks like physics: reading qvel from the wrong index.

A controller that reads qvel[7:] instead of qvel[6:] gets eleven joint
velocities plus one angular rate, shifted by one. It runs. It produces twelve
plausible numbers. And it is wrong in a way no tracking error will show you.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
# give the robot some motion so the arrays are not all zeros
d.qvel[:] = np.linspace(0.1, 1.8, m.nv)
mujoco.mj_forward(m, d)

right = d.qvel[6:]
wrong = d.qvel[7:]
print("qvel[6:]  (correct) first 4: %s" % np.round(right[:4], 3))
print("qvel[7:]  (wrong)   first 4: %s" % np.round(wrong[:4], 3))
print()
print("the wrong slice is short by one and shifted by one")
print("len correct %d, len wrong %d" % (len(right), len(wrong)))
print("max difference across the overlap: %.3f rad/s"
      % np.abs(right[:len(wrong)] - wrong).max())

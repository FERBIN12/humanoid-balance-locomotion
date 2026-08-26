#!/usr/bin/env python3
"""Why nq is 19 and nv is 18."""
import os, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
print("nq %d = 3 position + 4 quaternion + 12 joints" % m.nq)
print("nv %d = 3 linear   + 3 angular    + 12 joints" % m.nv)
print()
print("the difference is the quaternion: 4 numbers, 3 degrees of freedom")
print("so the joint slices are qpos[7:] and qvel[6:], never both 7")

#!/usr/bin/env python3
"""The caching trap: read a position before asking for it and you get the old one."""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
before = d.xpos[k].copy()

j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_hip_pitch_joint")
d.qpos[m.jnt_qposadr[j]] = -0.9        # move the hip a long way

stale = d.xpos[k].copy()               # read WITHOUT forward kinematics
mujoco.mj_forward(m, d)
fresh = d.xpos[k].copy()

print("knee position before      %s" % np.round(before, 4))
print("after setting the hip,")
print("  read without mj_forward %s   <- the OLD answer" % np.round(stale, 4))
print("  read after mj_forward   %s   <- correct" % np.round(fresh, 4))
print()
print("moved %.4f m, and the stale read showed %.4f m"
      % (np.linalg.norm(fresh - before), np.linalg.norm(stale - before)))

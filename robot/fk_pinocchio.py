#!/usr/bin/env python3
"""Forward kinematics with pinocchio, and why the root joint choice matters.

Run with PYTHONNOUSERSITE=1: the ROS build of pinocchio is compiled against
the system NumPy and crashes against a newer one in ~/.local.
"""
import os, numpy as np, pinocchio as pin

URDF = os.path.expanduser(
    "~/humanoid_ws/install/cortex_humanoid_description/share/"
    "cortex_humanoid_description/urdf/cortex_humanoid_hands.urdf")

# FIXED base: the pelvis is welded to the world
fixed = pin.buildModelFromUrdf(URDF)
# FREE FLYER base: the pelvis floats, which is what a walking robot needs
free = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
print("fixed base : nq %2d  nv %2d" % (fixed.nq, fixed.nv))
print("free flyer : nq %2d  nv %2d   <- use this one" % (free.nq, free.nv))
print("the extra 7 and 6 are the floating base; nq-nv=1 is the quaternion")
print()

d = free.createData()
q = pin.neutral(free)
q[2] = 1.02
pin.forwardKinematics(free, d, q)
pin.updateFramePlacements(free, d)      # two calls, in this order
print("frames in the model: %d" % free.nframes)
for f in ("pelvis", "left_ankle_roll_link", "right_ankle_roll_link"):
    i = free.getFrameId(f)
    print("  %-24s %s" % (f, np.round(d.oMf[i].translation, 4)))
y = d.oMf[free.getFrameId("left_ankle_roll_link")].translation[1]
print()
print("stance width = 2 x %.3f = %.3f m" % (abs(y), 2 * abs(y)))

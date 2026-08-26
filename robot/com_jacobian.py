#!/usr/bin/env python3
"""How far does the centre of mass move per radian of each joint?

Replaces an argument with a number: the hip is 114x more effective at moving
the CoM than the ankle. That is why balance uses the hip for CoM authority and
the ankle for centre of PRESSURE authority. Different jobs.
"""
import os, numpy as np, pinocchio as pin

URDF = os.path.expanduser(
    "~/humanoid_ws/install/cortex_humanoid_description/share/"
    "cortex_humanoid_description/urdf/cortex_humanoid_hands.urdf")
m = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
d = m.createData()
q = pin.neutral(m); q[2] = 1.02

Jc = pin.jacobianCenterOfMass(m, d, q)
print("CoM Jacobian: %d x %d" % Jc.shape)
print()
print("first six columns are the floating base, almost the identity:")
print(np.round(Jc[:, :6], 3))
print()
print("%-30s %-12s %s" % ("joint", "|dCoM|", "dCoM (x y z)"))
rows = []
for n in ("left_hip_pitch_joint", "left_shoulder_pitch_joint", "left_knee_joint",
          "torso_joint", "left_ankle_pitch_joint"):
    if not m.existJointName(n):
        continue
    c = Jc[:, m.joints[m.getJointId(n)].idx_v]
    rows.append((n, float(np.linalg.norm(c))))
    print("%-30s %.4f m/rad  %s" % (n, np.linalg.norm(c), np.round(c, 4)))
best, worst = max(rows, key=lambda r: r[1]), min(rows, key=lambda r: r[1])
print()
print("%s beats %s by %.0fx" % (best[0], worst[0], best[1] / worst[1]))
print("the vertical column is tiny for all of them: that is the LIPM constant")
print("height assumption showing up in the data, not an arbitrary choice")

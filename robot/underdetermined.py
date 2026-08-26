#!/usr/bin/env python3
"""3 x 57 is massively underdetermined, and that is a gift, not a problem."""
import os, numpy as np, pinocchio as pin

URDF = os.path.expanduser(
    "~/humanoid_ws/install/cortex_humanoid_description/share/"
    "cortex_humanoid_description/urdf/cortex_humanoid_hands.urdf")
m = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
d = m.createData()
q = pin.neutral(m); q[2] = 1.02
Jc = pin.jacobianCenterOfMass(m, d, q)

want = np.array([0.05, 0.0, 0.0])          # move the CoM 5 cm/s forward
print("desired CoM velocity: %s m/s" % want)
print()
# minimum norm solution
v1 = np.linalg.pinv(Jc) @ want
print("minimum norm solution: |v| = %.4f rad/s over %d joints"
      % (np.linalg.norm(v1), m.nv))
print("  achieves %s" % np.round(Jc @ v1, 4))

# a DIFFERENT solution: add anything in the null space and the CoM is unchanged
ns = np.eye(m.nv) - np.linalg.pinv(Jc) @ Jc
extra = ns @ np.random.default_rng(0).normal(size=m.nv)
v2 = v1 + extra * 0.5
print()
print("add a null space motion: |v| = %.4f rad/s" % np.linalg.norm(v2))
print("  achieves %s   <- the SAME CoM velocity" % np.round(Jc @ v2, 4))
print()
print("null space dimension: %d" % (m.nv - np.linalg.matrix_rank(Jc)))
print("that is %d degrees of freedom to spend on other objectives:" % (m.nv - 3))
print("keeping the feet planted, the torso upright, the joints inside limits")

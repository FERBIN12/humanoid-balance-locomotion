#!/usr/bin/env python3
"""The foot Jacobian: 6 x nv, rank 6, and mostly zeros."""
import os, numpy as np, pinocchio as pin

URDF = os.path.expanduser(
    "~/humanoid_ws/install/cortex_humanoid_description/share/"
    "cortex_humanoid_description/urdf/cortex_humanoid_hands.urdf")
m = pin.buildModelFromUrdf(URDF, pin.JointModelFreeFlyer())
d = m.createData()
q = pin.neutral(m); q[2] = 1.02

fid = m.getFrameId("left_ankle_roll_link")
pin.computeJointJacobians(m, d, q)
pin.updateFramePlacements(m, d)
J = pin.getFrameJacobian(m, d, fid, pin.LOCAL_WORLD_ALIGNED)

print("left foot Jacobian: %d x %d" % J.shape)
print("rank %d  (6 means every foot velocity direction is available)"
      % np.linalg.matrix_rank(J))
nz = [i for i in range(m.nv) if np.linalg.norm(J[:, i]) > 1e-9]
print("non-zero columns: %d of %d" % (len(nz), m.nv))
print("  the 6 base DOF plus the 6 joints of the LEFT leg, nothing else")
print()
# SINGULARITY. A floating base model always has rank 6 at the foot, because the
# base alone can move it. The singularity lives in the LEG subchain, so measure
# the six leg columns on their own. My first attempt measured the full matrix and
# reported that a straight knee was BETTER conditioned, which is nonsense.
leg = [m.joints[m.getJointId(n)].idx_v for n in
       ("left_hip_yaw_joint", "left_hip_pitch_joint", "left_hip_roll_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint")]
print("knee angle   smallest singular value of the 6 leg columns   rank")
for k in (0.0, 0.05, 0.15, 0.30, 0.60, 1.00):
    qk = q.copy()
    qk[m.joints[m.getJointId("left_knee_joint")].idx_q] = k
    pin.computeJointJacobians(m, d, qk)
    pin.updateFramePlacements(m, d)
    Jk = pin.getFrameJacobian(m, d, fid, pin.LOCAL_WORLD_ALIGNED)[:, leg]
    sv = np.linalg.svd(Jk, compute_uv=False)
    print("  %.2f rad                  %.6f                     %d"
          % (k, sv[-1], np.linalg.matrix_rank(Jk, tol=1e-6)))
print()
print("straight knee: rank 5 and a singular value of exactly zero.")
print("one direction of foot motion is UNAVAILABLE, and inverting that Jacobian")
print("asks for an infinite joint velocity. That is why a gait keeps knee flexion.")

#!/usr/bin/env python3
"""Hold every joint at its neutral angle, let go of the body, and watch.

This is the honest baseline for the whole project. A position controller that
tracks every joint perfectly is NOT a balance controller, and this measures
exactly how it fails.

The gains here are the manufacturer's, not tuned by me: whatever happens is a
property of position control, not of a gain I chose to make a point.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)

# manufacturer gains for the 12 leg joints
KP = np.array([200., 200., 200., 300., 60., 40.] * 2)
KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2)
NEUTRAL = np.zeros(m.nu)

mujoco.mj_forward(m, d)
z0 = float(d.qpos[2])
print("spawn pelvis height %.3f m" % z0)
print("holding all %d joints at neutral with the manufacturer's gains" % m.nu)
print()
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
FOOT_HALF = 0.120                 # half of a 0.240 m foot


def pitch_deg():
    """Torso pitch. Without this the script cannot tell sinking from falling."""
    R = d.xmat[1].reshape(3, 3)
    return float(np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))))


def com_past_foot():
    """CoM distance beyond the foot centre. Measure against the FEET: a
    CoM-to-pelvis distance stays small however far the body pitches over,
    because the pelvis falls with it."""
    mid = (d.xpos[FL] + d.xpos[FR]) / 2.0
    return float(d.subtree_com[0][0] - mid[0])


print("%6s %9s %9s %9s %8s %9s"
      % ("t", "pelvis_z", "worst_err", "foot_z", "pitch", "com-foot"))

lo = z0
esc = None                        # when the CoM first left the support polygon
for step in range(3000):          # 6 s at 2 ms
    tau = KP * (NEUTRAL - d.qpos[7:]) - KD * d.qvel[6:]
    d.ctrl[:] = tau
    mujoco.mj_step(m, d)
    lo = min(lo, float(d.qpos[2]))
    if esc is None and abs(com_past_foot()) > FOOT_HALF:
        esc = float(d.time)
    if step % 500 == 0 or step == 2999:
        err = float(np.abs(NEUTRAL - d.qpos[7:]).max())
        fz = float(d.xpos[mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")][2])
        print("%6.2f %9.3f %9.3f %9.3f %8.1f %9.3f"
              % (d.time, d.qpos[2], err, fz, pitch_deg(), com_past_foot()))

print()
print("started at %.3f m, ended at %.3f m, lowest %.3f m" % (z0, d.qpos[2], lo))
# MEASURE the verdict, never assert it. This script used to print "it did not
# topple: it SANK" unconditionally. It was wrong: the torso ends at 65 degrees
# with the CoM 0.85 m outside a 0.12 m foot, having travelled about a metre.
# The lesson is better than the one I claimed, because the joints DID track.
print("torso pitch at the end: %.1f deg" % pitch_deg())
print("CoM beyond the foot:    %.3f m  (foot half length %.3f m)"
      % (com_past_foot(), FOOT_HALF))
if esc is not None:
    print("the CoM left the support polygon at t = %.2f s" % esc)
    print("so it did NOT just sink: it FELL, and the joints tracked the whole")
    print("way down. Position control held every angle and still lost the body.")
else:
    print("the CoM never left the support polygon: this is a sink, not a fall.")
err = float(np.abs(NEUTRAL - d.qpos[7:]).max())
print("worst joint error at the end: %.3f rad = %.1f deg" % (err, np.degrees(err)))

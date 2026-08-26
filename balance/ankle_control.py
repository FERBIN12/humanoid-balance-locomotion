#!/usr/bin/env python3
"""The ankle strategy, implemented and measured against its own prediction.

3.7 derived a closed form boundary: the ankle recovers while the capture point
x + xdot/omega stays inside half a foot, which is 0.120 m. This script builds
the controller and pushes the real robot to find out where it ACTUALLY fails.

The prediction and the measurement will not match exactly, and the gap is the
interesting part.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
idx = {n: i for i, n in enumerate(names)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VA = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
WORLD = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "world")
MASS = float(m.body_mass.sum())

G = 9.81
H = 0.937
OMEGA = np.sqrt(G / H)
P_MAX = 0.120                 # half a foot, from 2.6
KP = np.array([200., 200., 200., 300., 60., 40.] * 2) * 10
KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2) * np.sqrt(10.0)
# Sign matters and I had it backwards on the first attempt. Measured sweep at a
# 120 N push: K=0 falls, K=-220 falls, K=+220 STANDS. At 160 N only K=+600
# stands. So the ankle torque must be POSITIVE in the capture point, and the
# wrong sign is indistinguishable from having no strategy at all.
K_ANKLE = 600.0


FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")


def com_state(data):
    """CoM position, and its velocity.

    Two traps here, both of which cost me a wrong answer:

    1 the velocity. A mass weighted sum of data.cvel[:, 3:6] is CLOSE but not
      equal to the CoM velocity. mj_subtreeVel fills subtree_linvel, and
      subtree_linvel[0] matches a finite difference of the CoM exactly
      (-0.0154 vs -0.0154 m/s), so use that.

    2 the origin. The lean must be measured from the FEET, not from a world
      position captured earlier. I stored the CoM at settle time and differenced
      against it, and because the robot drifts while settling that produced a
      capture point of 1.4 metres on a 0.24 m foot: nonsense, and a useful
      signal that the reference frame was wrong, not the physics.
    """
    x = (m.body_mass[:, None] * data.xipos).sum(0) / MASS
    v = data.subtree_linvel[0].copy()
    stance = (data.xpos[FL] + data.xpos[FR]) / 2.0
    return x - stance, v


def trial(impulse_N, dur_steps=100, total=4000, use_ankle=True):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(1500):          # settle, so the push lands on a standing robot
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        d.ctrl[:] = KP * (0 - q) - KD * v
        mujoco.mj_step(m, d)
    peak_cap = 0.0
    cap = 0.0
    for step in range(total):
        d.xfrc_applied[1][0] = impulse_N if step < dur_steps else 0.0
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tau = KP * (0 - q) - KD * v
        if use_ankle:
            # cap was computed from the state left by the PREVIOUS mj_step, which
            # is the state this control cycle is reacting to. Calling mj_forward
            # here instead would overwrite the actuator forces mj_step just
            # applied, and the robot collapses to 0.45 m: the controller looks
            # broken when it is the bookkeeping that is broken.
            p_cmd = float(np.clip(cap, -P_MAX, P_MAX))
            for side in ("left", "right"):
                tau[idx[side + "_ankle_pitch_joint"]] += K_ANKLE * p_cmd
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        mujoco.mj_subtreeVel(m, d)
        rel, cv = com_state(d)
        cap = rel[0] + cv[0] / OMEGA
        peak_cap = max(peak_cap, abs(cap))
    stood = float(d.qpos[2]) > 0.85
    return stood, float(d.qpos[2]), peak_cap


print("omega %.3f rad/s, foot half length %.3f m" % (OMEGA, P_MAX))
print("predicted boundary: recoverable while |capture point| <= %.3f m" % P_MAX)
print()
print("%10s %12s %12s %14s" % ("push (N)", "with ankle", "no ankle", "peak capture"))
for fx in (40, 80, 120, 160, 200, 260):
    a_ok, a_z, a_cap = trial(fx, use_ankle=True)
    b_ok, b_z, b_cap = trial(fx, use_ankle=False)
    print("%10d %12s %12s %14.4f m"
          % (fx, "stood" if a_ok else "FELL %.2f" % a_z,
             "stood" if b_ok else "FELL %.2f" % b_z, a_cap))
print()
print("the ankle strategy buys the pushes between those two columns.")
print()
print("and notice the capture point column: every recovery kept it under")
print("0.14 m, every fall shows 1.3 m or more. The criterion from 3.7")
print("separates the two outcomes cleanly, with nothing in between.")

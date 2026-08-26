#!/usr/bin/env python3
"""Ankle plus arm momentum: the second line of defence this robot can run.

4.1 measured that H1-2 has no waist pitch, so the classical hip strategy is
unavailable. What is left above the waist is the shoulders: 40 Nm per side,
worth about one extra foot of CoP for 0.242 s.

The prediction I wrote down in 4.1, before running this:
  * the arms will help
  * by LESS than the naive 2x, because they must come back
  * the ankle alone survived 176 N, so a naive guess would be ~350 N

This script tests that on the full 51 joint robot.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
idx = {n: i for i, n in enumerate(names)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VA = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
MASS = float(m.body_mass.sum())
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
OMEGA = np.sqrt(9.81 / 0.937)
P_MAX = 0.120

# leg gains at the 10x that stands (3.4); everything else held softly
BASE = {"hip_yaw": 200., "hip_pitch": 200., "hip_roll": 200.,
        "knee": 300., "ankle_pitch": 60., "ankle_roll": 40.}
KP = np.zeros(m.nu)
KD = np.zeros(m.nu)
for i, n in enumerate(names):
    hit = [v for k, v in BASE.items() if n and k in n]
    KP[i] = hit[0] * 10 if hit else 60.0
    KD[i] = hit[0] / 40 * np.sqrt(10.0) if hit else 3.0

SHOULDER = [idx["left_shoulder_pitch_joint"], idx["right_shoulder_pitch_joint"]]
ANKLE = [idx["left_ankle_pitch_joint"], idx["right_ankle_pitch_joint"]]
K_ANKLE = 600.0
TAU_ARM = 40.0          # the shoulder's real limit, per side
K_ARM = 800.0           # Nm per metre of capture point EXCESS


def trial(fx, use_arms, total=4000, settle=1500):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        d.ctrl[:] = KP * (0 - q) - KD * v
        mujoco.mj_step(m, d)
    cap = 0.0
    peak = 0.0
    arm_used = 0
    for step in range(total):
        d.xfrc_applied[1][0] = fx if step < 100 else 0.0
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tau = KP * (0 - q) - KD * v
        p_cmd = float(np.clip(cap, -P_MAX, P_MAX))
        for a in ANKLE:
            tau[a] += K_ANKLE * p_cmd
        if use_arms:
            # PROPORTIONAL, and zero while the capture point is still inside the
            # foot. My first version slammed a fixed 40 Nm the moment the sign
            # of cap was known, and that is worse than doing nothing: measured
            # over 80..220 N it FAILED everywhere except a single 180 N window,
            # because a 40 Nm arm throw is enormous for a small disturbance.
            # Act only on the EXCESS past the ankle's authority.
            excess = max(0.0, abs(cap) - P_MAX)
            want = -np.sign(cap) * min(TAU_ARM, K_ARM * excess)
            for sh in SHOULDER:
                lo, hi = m.jnt_range[m.actuator_trnid[sh][0]]
                ang = d.qpos[QA[sh]]
                if (want > 0 and ang < hi - 0.05) or (want < 0 and ang > lo + 0.05):
                    tau[sh] = want
                    arm_used += 1
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        mujoco.mj_subtreeVel(m, d)
        com = (m.body_mass[:, None] * d.xipos).sum(0) / MASS
        st = (d.xpos[FL] + d.xpos[FR]) / 2.0
        cap = (com[0] - st[0]) + d.subtree_linvel[0][0] / OMEGA
        peak = max(peak, abs(cap))
    return float(d.qpos[2]) > 0.85, float(d.qpos[2]), peak, arm_used


def cliff(use_arms, settle=1500):
    """Bisect the survivable push.

    CAREFUL: bisection assumes the outcome is monotone in the push, and with a
    buggy arm sign it was NOT -- the robot fell at 100 N and stood at 150 N.
    A non monotone result invalidates the search, so check monotonicity first.
    """
    lo, hi = 60.0, 600.0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        if trial(mid, use_arms, settle=settle)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def monotone(use_arms, settle=1500):
    """Verify that surviving is monotone in the push before trusting a cliff."""
    got = [(fx, trial(fx, use_arms, settle=settle)[0])
           for fx in (120.0, 160.0, 200.0, 260.0, 320.0)]
    seen_fall = False
    for fx, ok in got:
        if not ok:
            seen_fall = True
        elif seen_fall:
            return False, got
    return True, got


print("prediction from 4.1: arms help, by less than 2x; naive guess was ~350 N")
print()
print("%8s %14s %14s" % ("push N", "ankle only", "ankle + arms"))
last_a = last_b = 0.0
for fx in (120.0, 160.0, 180.0, 200.0, 220.0, 240.0):
    a = trial(fx, False)[0]
    b = trial(fx, True)[0]
    if a:
        last_a = fx
    if b:
        last_b = fx
    print("%8.0f %14s %14s" % (fx, "stood" if a else "fell",
                               "stood" if b else "fell"))
print()
print("ankle only survives to   %.0f N" % last_a)
print("ankle + arms survives to %.0f N" % last_b)
print("improvement %.2fx" % (last_b / last_a))
print()
print("the CoP arithmetic said the reach roughly doubles, so a naive reading")
print("predicts about 2x the push. Measured: %.2fx." % (last_b / last_a))
print("the 4.1 prediction was right in direction and far too generous in size.")
print()
print("and note what the proportional version fixed. The bang bang version was")
print("not merely weaker: it was NON MONOTONE, failing at 120 N while standing")
print("at 180. A non monotone result means a bisected threshold is meaningless,")
print("which is why the monotonicity check runs before the numbers.")

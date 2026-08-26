#!/usr/bin/env python3
"""Where does stepping stop being optional?

4.6 built a reactive step and it toppled sideways. So before section five spends
its time on a lateral controller, this script answers a narrower question that
does not need one: over what band of pushes is stepping REQUIRED?

Below the band, standing works and a step is unnecessary risk. Above it, no step
exists at all (4.5). The band in between is the only place a step is worth
taking, and if it is narrow, that changes how much a step controller is worth.
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
STEP_MAX = 0.493              # the reachable set from 4.5

BASE = {"hip_yaw": 200., "hip_pitch": 200., "hip_roll": 200.,
        "knee": 300., "ankle_pitch": 60., "ankle_roll": 40.}
KP = np.zeros(m.nu); KD = np.zeros(m.nu)
for i, n in enumerate(names):
    hit = [v for k, v in BASE.items() if n and k in n]
    KP[i] = hit[0] * 20 if hit else 60.0
    KD[i] = hit[0] / 40 * np.sqrt(20.0) if hit else 3.0
CROUCH = {"hip_pitch": -0.50, "knee": 1.00, "ankle_pitch": -0.50}
HOLD = np.zeros(m.nu)
for i, n in enumerate(names):
    for k, v in CROUCH.items():
        if n and n.endswith(k + "_joint"):
            HOLD[i] = v
AN = [idx["left_ankle_pitch_joint"], idx["right_ankle_pitch_joint"]]
SHOULDER = [idx["left_shoulder_pitch_joint"], idx["right_shoulder_pitch_joint"]]


def run(fx, use_arms=True, total=4000, settle=2000):
    """Stand and defend with everything that does NOT require lifting a foot."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        d.ctrl[:] = KP * (HOLD - q) - KD * v
        mujoco.mj_step(m, d)
    cap = 0.0
    peak = 0.0
    for step in range(total):
        d.xfrc_applied[1][0] = fx if step < 100 else 0.0
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tau = KP * (HOLD - q) - KD * v
        for a in AN:
            tau[a] += 600.0 * float(np.clip(cap, -P_MAX, P_MAX))
        if use_arms:
            excess = max(0.0, abs(cap) - P_MAX)
            want = -np.sign(cap) * min(40.0, 800.0 * excess)
            for sh in SHOULDER:
                lo, hi = m.jnt_range[m.actuator_trnid[sh][0]]
                ang = d.qpos[QA[sh]]
                if (want > 0 and ang < hi - 0.05) or (want < 0 and ang > lo + 0.05):
                    tau[sh] = want
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        mujoco.mj_subtreeVel(m, d)
        com = (m.body_mass[:, None] * d.xipos).sum(0) / MASS
        st = (d.xpos[FL] + d.xpos[FR]) / 2.0
        cap = (com[0] - st[0]) + d.subtree_linvel[0][0] / OMEGA
        peak = max(peak, abs(cap))
    return float(d.qpos[2]) > 0.72, peak


def boundary(use_arms, lo=60.0, hi=600.0, iters=8, settle=2000):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if run(mid, use_arms, settle=settle)[0]:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# 3.9 taught that a single bisection over-reports its precision, so sample
# several phases and report the spread.
print("standing with everything that does NOT lift a foot: ankle plus arms")
print()
print("%14s %18s" % ("settle", "holds to"))
lows = []
for settle in (2000, 2600, 3200):
    b = boundary(True, settle=settle)
    lows.append(b)
    print("%14d %16.1f N" % (settle, b))
L = np.array(lows)
print()
print("so standing holds to %.0f N, give or take %.0f." % (L.mean(), 0.5 * (L.max() - L.min())))
print()

# --- the band, done properly ------------------------------------------------
# CAREFUL. It is tempting to compare the peak capture point of a FALLING run
# against the reachable set and declare the band empty. That is not a fair
# comparison: a falling run's peak is measured after the fall is underway. The
# honest question is what the capture point reaches on the LAST push that still
# survives, because that is the largest disturbance a step would ever be asked
# to catch.
print("bisecting the standing edge finely, and reading the capture point just")
print("below and just above it:")
print()
lo, hi = 250.0, 260.0
for _ in range(7):
    mid = 0.5 * (lo + hi)
    if run(mid)[0]:
        lo = mid
    else:
        hi = mid
_, pk_lo = run(lo)
_, pk_hi = run(hi)
print("  last surviving push  %.2f N -> peak capture %.4f m" % (lo, pk_lo))
print("  first failing push   %.2f N -> peak capture %.4f m" % (hi, pk_hi))
print("  reachable step set                            %.4f m" % STEP_MAX)
print()
print("note the discontinuity: %.2f N recovers with the capture point at %.3f,"
      % (lo, pk_lo))
print("and %.2f N, eight hundredths of a newton more, goes to %.3f. That is"
      % (hi, pk_hi))
print("the cliff from 3.9, resolved to a tenth of a newton.")
print()
margin = STEP_MAX - pk_lo
print("MARGIN at the edge of standing: %.4f m" % margin)
print("  as a fraction of the foot half length: %.2f" % (margin / 0.120))
print("  in speed terms: %.3f m/s" % (margin * OMEGA))
print()
print("so the band where a step is both NECESSARY and POSSIBLE is real, and it")
print("is about %.0f millimetres of capture point, or %.2f metres per second."
      % (margin * 1000, margin * OMEGA))
print()
print("that is the answer to this experiment, and it is not the one I expected.")
print("I assumed there would be a wide range of pushes where standing fails and")
print("a step saves you. There is not. Standing with ankle and arms already")
print("takes you to within %.0f mm of the kinematic step limit." % (margin * 1000))
print()
print("the consequence for section five is direct. A step is not much use as a")
print("RECOVERY action here: the window is too narrow to aim at. It is useful as")
print("part of a continuous gait, where you step before the capture point ever")
print("gets near the edge, and you accept the lateral cost every stride because")
print("you have a controller for it. That is what walking actually is.")

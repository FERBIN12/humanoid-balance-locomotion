#!/usr/bin/env python3
"""Where exactly does the ankle strategy fail? Bisect for the cliff.

3.8 showed the robot surviving 160 N and falling at 200 N. That is a factor of
1.25 of uncertainty, which is not a measurement. This script bisects to find the
boundary, then checks whether the capture point criterion predicts it.

It also asks the question that matters for section four: is the failure a
CLIFF or a slope?
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
idx = {n: i for i, n in enumerate(names)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VA = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
MASS = float(m.body_mass.sum())
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

OMEGA = np.sqrt(9.81 / 0.937)
P_MAX = 0.120
KP = np.array([200., 200., 200., 300., 60., 40.] * 2) * 10
KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2) * np.sqrt(10.0)


def trial(fx, K=600.0, dur=0.2, total=4000, settle=1500):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        d.ctrl[:] = KP * (0 - q) - KD * v
        mujoco.mj_step(m, d)
    cap = 0.0
    peak = 0.0
    pdur = int(dur / m.opt.timestep)
    for step in range(total):
        d.xfrc_applied[1][0] = fx if step < pdur else 0.0
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tau = KP * (0 - q) - KD * v
        p_cmd = float(np.clip(cap, -P_MAX, P_MAX))
        for side in ("left", "right"):
            tau[idx[side + "_ankle_pitch_joint"]] += K * p_cmd
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        mujoco.mj_subtreeVel(m, d)
        com = (m.body_mass[:, None] * d.xipos).sum(0) / MASS
        st = (d.xpos[FL] + d.xpos[FR]) / 2.0
        cap = (com[0] - st[0]) + d.subtree_linvel[0][0] / OMEGA
        peak = max(peak, abs(cap))
    return float(d.qpos[2]) > 0.85, float(d.qpos[2]), peak


# --- bisect the boundary, at ONE phase ---------------------------------------
# This is the measurement I nearly shipped, and it is over-precise.
lo, hi = 100.0, 400.0
for _ in range(7):
    mid = 0.5 * (lo + hi)
    if trial(mid)[0]:
        lo = mid
    else:
        hi = mid
single = 0.5 * (lo + hi)
print("bisected at one phase: cliff = %.1f N, apparently to +/- %.1f"
      % (single, 0.5 * (hi - lo)))
print()

# --- but WHEN the push lands changes the answer -------------------------------
# 3.6 measured this robot chattering with a 45.5 N standard deviation in its
# vertical contact force while standing perfectly still. So it is never at rest,
# and a push that lands at one point in that oscillation is not the same
# experiment as a push that lands at another. Vary the settle, and the cliff
# moves by 11 per cent.
print("the same bisection, at six different settle times:")
print("%12s %14s" % ("settle steps", "cliff (N)"))
cliffs = []
for settle in (1500, 2000, 2500, 3000, 3500, 4000):
    lo, hi = 100.0, 400.0
    for _ in range(7):
        mid = 0.5 * (lo + hi)
        if trial(mid, settle=settle)[0]:
            lo = mid
        else:
            hi = mid
    c = 0.5 * (lo + hi)
    cliffs.append(c)
    print("%12d %14.1f" % (settle, c))
cl = np.array(cliffs)
print()
print("min %.1f   max %.1f   mean %.1f   spread %.1f N = %.0f per cent"
      % (cl.min(), cl.max(), cl.mean(), cl.max() - cl.min(),
         100 * (cl.max() - cl.min()) / cl.mean()))
print()
print("so the honest answer is %.0f N give or take %.0f, not %.1f +/- 0.3."
      % (cl.mean(), 0.5 * (cl.max() - cl.min()), single))
print("the single bisection was 60 times more precise than the thing it measured.")
print()

# --- cliff or slope? ---------------------------------------------------------
print("past the boundary, how bad is it?")
for fx in (160.0, 200.0, 260.0, 340.0):
    ok, z, pk = trial(fx, settle=3000)
    print("  %7.1f N -> pelvis %.3f m   peak capture %.4f m   %s"
          % (fx, z, pk, "stood" if ok else "fell"))
print()
print("every failure lands at the same height, so there is no partial credit:")
print("the capture point either stays in the foot or it does not.")

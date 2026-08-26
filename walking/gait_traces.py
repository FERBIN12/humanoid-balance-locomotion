#!/usr/bin/env python3
"""5.8 -- the four traces that tell you what broke.

5.7 fell and it took me a support polygon plot to see why. This builds the
instrument properly: four quantities, logged together, chosen so that each one
fails FIRST for a different class of bug. The point is not that four plots are
prettier than one. It is that a single trace cannot distinguish causes.

Run it against the 5.7 controller and against three deliberately broken
variants, and see which trace moves first in each case.
"""
import numpy as np
import mujoco, os

SCENE = os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
NAMES = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(m.nu)]
IDX = {n: i for i, n in enumerate(NAMES)}
QA = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
VA = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

KP0 = np.array([200., 200., 200., 300., 60., 40.] * 2)
KD0 = np.array([5., 5., 5., 7.5, 2., 2.] * 2)
T_STEP, STEP_L, CLEAR, HIP_H = 0.735, 0.30, 0.05, 0.780
LT = LS = 0.400
LA = 0.020
FOOT_HALF = 0.120
OMEGA = np.sqrt(9.81 / 0.937)


def ik(dx, dz):
    tz = dz + LA
    r = np.hypot(dx, tz)
    if r > LT + LS:
        return None
    c = (r * r - LT ** 2 - LS ** 2) / (2 * LT * LS)
    knee = np.arccos(np.clip(c, -1.0, 1.0))
    beta = np.arctan2(-dx, -tz)
    alpha = np.arctan2(LS * np.sin(knee), LT + LS * np.cos(knee))
    return beta - alpha, knee


def run(gain=10.0, step_l=STEP_L, clear=CLEAR, t_step=T_STEP,
        dur=6.0, settle=1.5):
    """Return the four traces. Every variant differs by ONE argument."""
    KP, KD = KP0 * gain, KD0 * np.sqrt(gain)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ns = int(settle / m.opt.timestep)
    out = {"t": [], "com": [], "track": [], "contact": [], "cap": []}
    for step in range(int(dur / m.opt.timestep)):
        t = step * m.opt.timestep
        tgt = np.zeros(m.nu)
        if step < ns:
            k = min(1.0, step / max(1, ns * 0.6))
            hp, kn = ik(0.0, -HIP_H)
            for side in ("left", "right"):
                tgt[IDX[side + "_hip_pitch_joint"]] = hp * k
                tgt[IDX[side + "_knee_joint"]] = kn * k
                tgt[IDX[side + "_ankle_pitch_joint"]] = -(hp + kn) * k
        else:
            gt = t - settle
            u = (gt % t_step) / t_step
            sw = "left" if int(gt / t_step) % 2 == 0 else "right"
            st = "right" if sw == "left" else "left"
            sx = step_l * 0.5 * (1 - np.cos(np.pi * u)) - step_l / 2
            sz = -HIP_H + clear * np.sin(np.pi * u)
            for side, (fx, fz) in ((sw, (sx, sz)), (st, (-sx, -HIP_H))):
                sol = ik(fx, fz)
                if sol is None:
                    continue
                hp, kn = sol
                tgt[IDX[side + "_hip_pitch_joint"]] = hp
                tgt[IDX[side + "_knee_joint"]] = kn
                tgt[IDX[side + "_ankle_pitch_joint"]] = -(hp + kn)
        q = d.qpos[QA]
        d.ctrl[:] = KP * (tgt - q) - KD * d.qvel[VA]
        mujoco.mj_step(m, d)
        if step % 20 == 0:
            mujoco.mj_subtreeVel(m, d)
            mid = (d.xpos[FL] + d.xpos[FR]) / 2.0
            com = d.subtree_com[0]
            vx = float(d.subtree_linvel[0][0])
            out["t"].append(t)
            # 1 balance: how far the CoM is outside the support polygon
            out["com"].append(float(np.hypot(com[0] - mid[0], com[1] - mid[1])))
            # 2 tracking: is the controller achieving the pose it commanded
            out["track"].append(float(np.abs(tgt - d.qpos[QA]).max()))
            # 3 contact: is the robot still touching the floor as expected
            out["contact"].append(int(d.ncon))
            # 4 capture point: where the momentum says it is heading
            out["cap"].append(float((com[0] - mid[0]) + vx / OMEGA))
    return {k: np.array(v) for k, v in out.items()}


def first_breach(tr, key, limit, after=0.0):
    """When did this trace first leave its healthy band, AFTER a given time?

    The `after` argument is not decoration. Measuring from t=0 reports SETTLE
    failures as gait failures: the soft gain case below never reaches the
    starting pose at all, and without this filter its three traces all breach
    during the crouch and the table says "everything failed at once", which
    diagnoses nothing.
    """
    i = tr["t"] >= after
    bad = np.abs(tr[key][i]) > limit
    if not bad.any():
        return None
    return float(tr["t"][i][np.argmax(bad)])


SETTLE = 1.5

print("--- the four traces, and what each one is FOR ---")
print("  1 CoM offset      balance: is the mass still over the feet")
print("  2 tracking error  actuation: is the controller achieving its pose")
print("  3 contact count   support: is the robot touching the floor at all")
print("  4 capture point   momentum: is a step still able to catch it")
print()

CASES = [
    ("5.7 as built", dict()),
    ("gain 1x, too soft", dict(gain=1.0)),
    ("step 0.60 m, too long", dict(step_l=0.60)),
    ("clearance 0.001 m", dict(clear=0.001)),
]

print("--- 1 first: is the run even valid? ---")
print("  A gait test that never reaches its starting pose is not a gait test.")
print("%26s %18s %10s" % ("case", "track at handover", "verdict"))
valid = {}
runs = {}
for label, kw in CASES:
    tr = run(**kw)
    runs[label] = tr
    i = int(np.searchsorted(tr["t"], SETTLE))
    e = float(tr["track"][i])
    ok = e < 0.35
    valid[label] = ok
    print("%26s %15.3f rad %10s" % (label, e, "valid" if ok else "INVALID"))
print()
print("  the soft gain case is not a slower failure, it is a different")
print("  experiment: at 0.842 rad it never got into the crouch the gait was")
print("  designed around. Reporting its trace order would be reporting noise.")
print()

print("--- 2 which trace moves first, measured after the step command ---")
print("%26s %11s %11s %11s" % ("case", "balance", "tracking", "capture"))
rows = []
for label, kw in CASES:
    if not valid[label]:
        print("%26s %11s %11s %11s" % (label, "-", "-", "-"))
        continue
    tr = runs[label]
    b = first_breach(tr, "com", FOOT_HALF, SETTLE)
    k = first_breach(tr, "track", 0.35, SETTLE)
    c = first_breach(tr, "cap", 0.493, SETTLE)
    rows.append((label, b, k, c))
    fmt = lambda v: ("%.2f s" % v) if v is not None else "never"
    print("%26s %11s %11s %11s" % (label, fmt(b), fmt(k), fmt(c)))
print()

print("--- 3 what the ORDER says ---")
for label, b, k, c in rows:
    order = sorted([(v, n) for v, n in ((b, "balance"), (k, "tracking"),
                                        (c, "capture")) if v is not None])
    if not order:
        print("  %-24s nothing breached" % label)
        continue
    lead = order[0]
    gap = (order[1][0] - lead[0]) if len(order) > 1 else None
    print("  %-24s %s first at %.2f s%s"
          % (label, lead[1], lead[0],
             (", next %.2f s later" % gap) if gap else ""))
print()
print("  In all three valid cases BALANCE moves first, and I want to be honest")
print("  that this is not what I expected to find. I assumed a long step would")
print("  show up as a tracking failure and a soft gain as a torque failure.")
print("  It does not: on this robot the mass leaves the feet before the joints")
print("  fall behind, whatever you break. That is a fact about the machine,")
print("  not about the instrument, and it means BALANCE is the trace to watch")
print("  if you only get one.")
print()
print("  What the other three buy you is the GAP. 5.7 as built has 0.40 s")
print("  between balance and tracking; the long step has 0.00 s. A fault that")
print("  breaks balance and tracking together is a fault in the PLAN. A fault")
print("  that breaks balance alone, with tracking following much later, is a")
print("  robot doing exactly what it was told and being told the wrong thing.")
print()

print("--- 4 contact, and a claim I had to withdraw ---")
print("%26s %8s %8s %8s" % ("case", "min", "max", "mean"))
for label, kw in CASES:
    tr = runs[label]
    post = tr["contact"][tr["t"] > SETTLE]
    print("%26s %8d %8d %8.1f" % (label, post.min(), post.max(), post.mean()))
print()
print("  I wrote that a scuffing foot would show up here and nowhere else.")
print("  The numbers say otherwise: the clearance 0.001 m case has the same")
print("  mean contact count as the baseline, 2.7, and the same min and max.")
print("  A 1 mm clearance simply does not scuff on flat ground, so there was")
print("  no scuff to detect. The trace is still worth logging, because zero")
print("  contacts means airborne and a stuck high count means a fallen robot,")
print("  but I do not have evidence for the claim I wanted to make about it.")

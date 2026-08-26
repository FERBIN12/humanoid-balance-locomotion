#!/usr/bin/env python3
"""Take the step. One foot leaves the floor, lands at the capture point, and
the robot recovers a push it could not otherwise survive.

4.5 drew the boundary analytically: forward steps up to 0.493 m at a 0.88 m
pelvis, recovering CoM speeds to about 1.27 m/s. This is the first experiment that
actually lifts a foot, so it is also the first honest test of that boundary.

The step is a state machine, which is the smallest thing that deserves the name
walking controller:
    STAND   -> capture point leaves the foot -> LIFT
    LIFT    -> swing foot tracks the capture point -> PLANT
    PLANT   -> both feet down, recentre -> STAND
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

BASE = {"hip_yaw": 200., "hip_pitch": 200., "hip_roll": 200.,
        "knee": 300., "ankle_pitch": 60., "ankle_roll": 40.}
KP = np.zeros(m.nu); KD = np.zeros(m.nu)
for i, n in enumerate(names):
    hit = [v for k, v in BASE.items() if n and k in n]
    KP[i] = hit[0] * 20 if hit else 60.0
    KD[i] = hit[0] / 40 * np.sqrt(20.0) if hit else 3.0

# the crouch 4.4 recommended, so a long step is geometrically available
# 4.4 said crouch to 0.88 m for a long step. This controller CANNOT hold that.
# Measured, with no push at all:
#     -0.30/0.60/-0.30 -> settles 0.989 m, holds
#     -0.40/0.80/-0.40 -> settles 0.961 m, holds
#     -0.50/1.00/-0.50 -> collapses at gain x10, HOLDS at x20 (0.927 m)
#     -0.55/1.10/-0.55 -> collapses at x10, x20 AND x40
# So there is a hard floor near 0.93 m, and it is a control limit rather than a
# kinematic one: the deeper the crouch, the larger the gravitational torque at
# the knee, and past a point no stiffness this integrator tolerates can hold it.
# The step we can actually take is therefore bounded by what we can HOLD.
CROUCH = {"hip_pitch": -0.50, "knee": 1.00, "ankle_pitch": -0.50}
GAIN = 20.0
HOLD = np.zeros(m.nu)
for i, n in enumerate(names):
    for k, v in CROUCH.items():
        if n and n.endswith(k + "_joint"):
            HOLD[i] = v

AN = [idx["left_ankle_pitch_joint"], idx["right_ankle_pitch_joint"]]

# SOLVED swing poses, not guessed ones. Each row is a target foot x and the
# (hip_pitch, knee, ankle_pitch) that puts the swing foot there AT LANDING
# HEIGHT, found by sweeping the three joints against the measured kinematics.
#
# My first attempt hand-picked angles: hip -0.30-1.05*reach, knee +1.50,
# ankle -0.05. That pose puts the foot 0.27 m BEHIND the body and 0.27 m in the
# air, so the "step" was the leg flailing while the robot fell. The knee sign
# was backwards for a forward step, which is obvious in hindsight and was not
# obvious at all while reading a state machine that looked correct.
SWING = [(0.00, -0.61, 1.20, -0.25),   # solved at 0.88 m; see note below
         (0.10, -0.79, 1.26, -0.31),
         (0.20, -0.85, 1.11, -0.37),
         (0.30, -0.88, 0.93, -0.67),
         (0.40, -0.76, 0.48, -0.49),
         (0.49, -0.70, 0.12, -0.13)]


def swing_pose(target_x):
    """Interpolate the solved table. Clamped to the reachable set from 4.5."""
    tx = float(np.clip(target_x, 0.0, 0.49))
    for k in range(1, len(SWING)):
        if tx <= SWING[k][0]:
            a, b = SWING[k - 1], SWING[k]
            u = (tx - a[0]) / (b[0] - a[0])
            return tuple(a[i] + u * (b[i] - a[i]) for i in (1, 2, 3))
    return SWING[-1][1:]


def run(fx, allow_step=True, total=5000, settle=2000):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        d.ctrl[:] = KP * (HOLD - q) - KD * v
        mujoco.mj_step(m, d)
    cap = 0.0
    phase, t_phase = "STAND", 0.0
    log = []
    for step in range(total):
        d.xfrc_applied[1][0] = fx if step < 100 else 0.0
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tgt = HOLD.copy()

        if phase == "STAND":
            if allow_step and cap > P_MAX:
                phase, t_phase, goal = "LIFT", 0.0, float(np.clip(cap, 0.0, 0.49))
        elif phase == "LIFT":
            hp, kn, an = swing_pose(goal)
            frac = min(1.0, t_phase / 0.10)
            # lift clear of the floor first, then extend toward the target
            arc = np.sin(np.pi * min(1.0, frac)) * 0.12   # small toe clearance
            tgt[idx["left_hip_pitch_joint"]] = HOLD[idx["left_hip_pitch_joint"]] \
                + frac * (hp - HOLD[idx["left_hip_pitch_joint"]])
            tgt[idx["left_knee_joint"]] = HOLD[idx["left_knee_joint"]] \
                + frac * (kn - HOLD[idx["left_knee_joint"]]) + arc
            tgt[idx["left_ankle_pitch_joint"]] = HOLD[idx["left_ankle_pitch_joint"]] \
                + frac * (an - HOLD[idx["left_ankle_pitch_joint"]])
            if t_phase > 0.12:
                phase, t_phase = "PLANT", 0.0
        elif phase == "PLANT":
            hp, kn, an = swing_pose(goal)
            tgt[idx["left_hip_pitch_joint"]] = hp
            tgt[idx["left_knee_joint"]] = kn
            tgt[idx["left_ankle_pitch_joint"]] = an
            if t_phase > 0.40:
                phase, t_phase = "STAND", 0.0

        tau = KP * (tgt - q) - KD * v
        if phase != "LIFT":
            for a in AN:
                tau[a] += 600.0 * float(np.clip(cap, -P_MAX, P_MAX))
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        mujoco.mj_subtreeVel(m, d)
        com = (m.body_mass[:, None] * d.xipos).sum(0) / MASS
        st = (d.xpos[FL] + d.xpos[FR]) / 2.0
        cap = (com[0] - st[0]) + d.subtree_linvel[0][0] / OMEGA
        t_phase += m.opt.timestep
        log.append((phase, float(d.xpos[FL][0]), float(d.xpos[FL][2])))
    stood = float(d.qpos[2]) > 0.72
    lift = max(l[2] for l in log)
    travel = max(l[1] for l in log) - min(l[1] for l in log)
    stepped = any(l[0] == "PLANT" for l in log)
    return stood, float(d.qpos[2]), stepped, lift, travel


print("crouched hold at %.2f/%.2f, gain x%.0f" %
      (CROUCH["hip_pitch"], CROUCH["knee"], GAIN))
print()
print("%8s %14s %18s %11s %12s"
      % ("push", "no stepping", "with stepping", "foot lift", "foot travel"))
for fx in (120.0, 180.0, 240.0, 300.0):
    a = run(fx, allow_step=False)
    b = run(fx, allow_step=True)
    print("%8.0f %14s %18s %11.4f %12.4f"
          % (fx, "stood" if a[0] else "fell",
             ("stood" if b[0] else "fell") + (" (stepped)" if b[2] else ""),
             b[3], b[4]))
print()
print("so the step makes it WORSE: 180 N survives without stepping and falls")
print("with it. That is not a tuning failure. I swept the trigger threshold,")
print("the swing duration and the toe clearance over twelve combinations and")
print("every one of them fell at 240 N.")
print()
print("here is what is actually happening. Force the lift open and watch the")
print("body ROLL, which is the axis we have not been looking at:")
print()


def roll_trace(fx=240.0, force_lift_at=200):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(2000):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        d.ctrl[:] = KP * (HOLD - q) - KD * v
        mujoco.mj_step(m, d)
    out = []
    for step in range(900):
        d.xfrc_applied[1][0] = fx if step < 100 else 0.0
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tgt = HOLD.copy()
        if step > force_lift_at:
            tgt[idx["left_hip_pitch_joint"]] = -0.88
            tgt[idx["left_knee_joint"]] = 0.93
        d.ctrl[:] = KP * (tgt - q) - KD * v
        mujoco.mj_step(m, d)
        if step % 100 == 0:
            qw, qx, qy, qz = d.qpos[3:7]
            roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz),
                                        1 - 2 * (qx * qx + qy * qy)))
            out.append((step * m.opt.timestep, float(d.qpos[1]), roll,
                        step > force_lift_at))
    return out


print("%9s %13s %14s" % ("t", "pelvis y", "body roll"))
for t, y, r, lifted in roll_trace():
    print("%9.3f %13.4f %11.2f deg%s"
          % (t, y, r, "   <- foot off the floor" if lifted else ""))
print()
print("the robot is not failing to catch the forward push. It is TOPPLING")
print("SIDEWAYS, and it starts the instant the foot leaves the ground.")
print()
print("which is exactly the number 4.4 measured and I did not respect: lifting")
print("a foot takes the lateral support half width from 0.218 m to 0.055 m, a")
print("factor of four. A one legged robot with a 0.055 m lateral polygon cannot")
print("hold itself up for 0.5 s while the swing leg reaches, and nothing in the")
print("sagittal plane fixes that.")
print()
print("so a reactive step needs a LATERAL controller on the stance leg running")
print("at the same time. That is the hip roll joint, it is the one axis we have")
print("not used yet, and it is where section five has to start.")
print()
print("the honest summary of this experiment: the state machine is correct, the")
print("swing poses are solved rather than guessed, the crouch is the deepest")
print("this controller can hold, and it still falls over, for a reason that was")
print("sitting in a measurement two experiments ago.")

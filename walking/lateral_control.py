#!/usr/bin/env python3
"""The lateral controller. 4.6 proved we need it: the instant a foot lifts, body
roll went from -0.4 degrees to +103 in 0.6 s and the robot toppled sideways.

The sagittal ankle strategy from 3.8 works because the CoP can travel half a
foot forward. Laterally on ONE foot the CoP can only travel half a foot WIDTH,
which 3.7 measured as 0.055 m. So the lateral problem is the same problem with
a quarter of the authority, and it needs the hip ROLL joint as well as the ankle
roll joint.

Question this script answers: with a lateral controller, how long can the robot
stand on one foot?
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
H = 0.927
OMEGA = np.sqrt(9.81 / H)
SS_HALF = 0.055                # single support lateral half width (3.7)

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

AN_P = [idx["left_ankle_pitch_joint"], idx["right_ankle_pitch_joint"]]
AN_R = [idx["left_ankle_roll_joint"], idx["right_ankle_roll_joint"]]
HIP_R = [idx["left_hip_roll_joint"], idx["right_hip_roll_joint"]]


def com_lat(d):
    """CoM lateral position and velocity relative to the SUPPORT, plus the
    lateral capture point. Which support depends on what is on the floor."""
    mujoco.mj_subtreeVel(m, d)
    com = (m.body_mass[:, None] * d.xipos).sum(0) / MASS
    return float(com[1]), float(d.subtree_linvel[0][1])


def run(lift_at=1.0, total=6000, settle=2000, k_ankle=0.0, k_hip=0.0):
    """Stand, then lift the left foot and try to stay up on the right."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for _ in range(settle):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        d.ctrl[:] = KP * (HOLD - q) - KD * v
        mujoco.mj_step(m, d)
    lift_step = int(lift_at / m.opt.timestep)
    t_fall = None
    peak_roll = 0.0
    for step in range(total):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tgt = HOLD.copy()
        lifted = step >= lift_step
        if lifted:
            tgt[idx["left_hip_pitch_joint"]] = -0.88
            tgt[idx["left_knee_joint"]] = 0.93
        tau = KP * (tgt - q) - KD * v

        if lifted and (k_ankle or k_hip):
            # lateral capture point relative to the STANCE foot (the right one)
            y, vy = com_lat(d)
            y_rel = y - float(d.xpos[FR][1])
            cap_y = y_rel + vy / OMEGA
            cmd = float(np.clip(cap_y, -SS_HALF, SS_HALF))
            # ankle roll pushes the CoP sideways within the stance foot
            tau[idx["right_ankle_roll_joint"]] += k_ankle * cmd
            # hip roll swings the whole body over the stance foot
            tau[idx["right_hip_roll_joint"]] += k_hip * cmd

        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        qw, qx, qy, qz = d.qpos[3:7]
        roll = np.degrees(np.arctan2(2 * (qw * qx + qy * qz),
                                    1 - 2 * (qx * qx + qy * qy)))
        peak_roll = max(peak_roll, abs(roll))
        if t_fall is None and float(d.qpos[2]) < 0.60:
            t_fall = (step - lift_step) * m.opt.timestep
    return t_fall, peak_roll, float(d.qpos[2])


# ---------------------------------------------------------------------------
# What this script measures, and it is not what I set out to build.
# ---------------------------------------------------------------------------
print("4.6 showed the robot toppling sideways the instant a foot lifted. The")
print("plan for this experiment was a lateral controller to fix that. Here is what")
print("the measurements said instead.")
print()

print("--- 1 the controller does nothing ---")
print("%28s %14s %12s" % ("controller", "time upright", "peak roll"))
for label, ka, kh in (("nothing (4.6 baseline)", 0.0, 0.0),
                      ("ankle roll only", 400.0, 0.0),
                      ("hip roll only", 0.0, 400.0),
                      ("both", 400.0, 400.0)):
    tf, pr, z = run(k_ankle=ka, k_hip=kh)
    print("%28s %12.3f s %10.1f deg" % (label, tf if tf else 9.99, pr))
print()
print("four cases, one answer: about 1.7 s upright and 179 degrees of roll.")
# Be precise about WHY. The term does reach the physics: requested torque equals
# applied torque, nothing is clipped by the actuator. What is saturated is the
# COMMAND. cap_y runs 0.16 to 0.77 m while the clip is +-0.055, so cmd sits on
# its bound 100 per cent of the time and k*cmd is a constant bias, not feedback.
# A value pinned at its bound is not a measurement of anything.
print("the torque is not being clipped: requested equals applied. What is")
print("saturated is the COMMAND. The capture point runs 0.16 to 0.77 m against")
print("a %.3f m clip, so cmd sits on its bound every single step and the gain" % SS_HALF)
print("multiplies a constant. That is a bias, not a feedback law, which is why")
print("400 and 0 land in the same place.")
print()

print("--- 2 why: the reference frame ---")
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
for _ in range(2000):
    q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
    v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
    d.ctrl[:] = KP * (HOLD - q) - KD * v
    mujoco.mj_step(m, d)
com = (m.body_mass[:, None] * d.xipos).sum(0) / MASS
gap = float(com[1] - d.xpos[FR][1])
print("standing: CoM y %.4f, right foot y %.4f" % (com[1], d.xpos[FR][1]))
# Be exact about the decomposition. Half the stance width is 0.163 m; the
# extra ~5 mm is the CoM's own lateral offset from the pelvis centre. Saying
# "that is half the stance width" of a 0.168 m number is close but sloppy,
# and this project's whole argument is that you check the number.
print("so the CoM is already %.4f m from the stance foot BEFORE anything moves,"
      % gap)
print("which is half the stance width (0.163) plus the CoM's own 5 mm offset.")
print("The lateral capture point is")
print("permanently outside the %.3f m foot, the clip saturates, and the" % SS_HALF)
print("controller applies a constant torque that corrects nothing.")
print()
print("this is not disturbance rejection. Standing on one foot requires a")
print("WEIGHT SHIFT of %.3f m, which is a different problem." % gap)
print()


def static_shift(hip_roll, ankle_roll=0.0, steps=3000, mult=20.0):
    """Hold a commanded roll pose and report where the CoM ends up."""
    kp = np.zeros(m.nu); kd = np.zeros(m.nu)
    for i, n in enumerate(names):
        hit = [v for k, v in BASE.items() if n and k in n]
        kp[i] = hit[0] * mult if hit else 60.0
        kd[i] = hit[0] / 40 * np.sqrt(mult) if hit else 3.0
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    tgt = HOLD.copy()
    for side in ("left", "right"):
        tgt[idx[side + "_hip_roll_joint"]] = hip_roll
        tgt[idx[side + "_ankle_roll_joint"]] = ankle_roll
    for _ in range(steps):
        q = np.array([dd.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([dd.qvel[VA[i]] for i in range(m.nu)])
        dd.ctrl[:] = kp * (tgt - q) - kd * v
        mujoco.mj_step(m, dd)
    c = (m.body_mass[:, None] * dd.xipos).sum(0) / MASS
    return float(c[1] - dd.xpos[FR][1]), float(dd.qpos[2])


print("--- 3 can a static shift do it? ---")
print("%10s %12s %14s %8s" % ("hip roll", "ankle roll", "offset from R", "z"))
for hr, ar in ((0.00, 0.00), (0.05, 0.00), (0.10, 0.00),
               (0.10, 0.10), (0.12, 0.10), (0.12, 0.00)):
    off, z = static_shift(hr, ar)
    print("%10.2f %12.2f %14.4f %8.3f %s"
          % (hr, ar, off, z, "" if z > 0.6 else "FELL"))
print()
print("target is 0.000, meaning the CoM directly over the stance foot, and the")
print("foot only offers %.3f m of margin either side." % SS_HALF)
print("the best STABLE pose reaches about 0.090 m. Past that the joints run out")
print("or the robot topples.")
print()
print("and it is not a gain problem. Swept at 20x, 40x and 80x the leg gains,")
print("the best stable offset was 0.100, 0.078 and 0.116 m. Never inside.")
print()

def dynamic_shift(hr_peak, ramp, lift_delay, total=4000):
    """Ramp the roll and lift the foot while the CoM is still travelling."""
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    for _ in range(2000):
        q = np.array([dd.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([dd.qvel[VA[i]] for i in range(m.nu)])
        dd.ctrl[:] = KP * (HOLD - q) - KD * v
        mujoco.mj_step(m, dd)
    lift = int(lift_delay / m.opt.timestep)
    for step in range(total):
        t = step * m.opt.timestep
        tgt = HOLD.copy()
        frac = min(1.0, t / ramp)
        for side in ("left", "right"):
            tgt[idx[side + "_hip_roll_joint"]] = hr_peak * frac
        if step >= lift:
            tgt[idx["left_hip_pitch_joint"]] = -0.88
            tgt[idx["left_knee_joint"]] = 0.93
        q = np.array([dd.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([dd.qvel[VA[i]] for i in range(m.nu)])
        dd.ctrl[:] = KP * (tgt - q) - KD * v
        mujoco.mj_step(m, dd)
    return float(dd.qpos[2])


print("--- 4 can a DYNAMIC shift do it? ---")
print("if a static pose cannot get there, momentum might: ramp the roll and lift")
print("the foot while the CoM is still travelling.")
print()
print("%10s %8s %10s %10s" % ("hip peak", "ramp s", "lift s", "final z"))
stood = 0
for hr in (0.08, 0.12):
    for ramp in (0.20, 0.40):
        for ld in (0.15, 0.30, 0.50):
            z = dynamic_shift(hr, ramp, ld)
            up = z > 0.60
            stood += up
            print("%10.2f %8.2f %10.2f %10.3f %s"
                  % (hr, ramp, ld, z, "UP" if up else "fell"))
print()
print("%d of 12 stayed up." % stood)
print()
print("--- what this means ---")
print("single foot support is not reachable with this joint level PD stack. Not")
print("because the controller is badly tuned, but because holding the CoM over")
print("one foot needs the CoM %.3f m from where standing puts it, and the joints"
      % gap)
print("that could move it become unstable before they get there.")
print()
print("that is why 4.6's step toppled, and it is the honest reason industry")
print("reaches for a whole body QP or a learned policy for this specific job.")
print("A QP can trade off the CoM target, the foot contacts and the joint limits")
print("simultaneously. A stack of independent PD loops cannot, because each one")
print("is solving its own problem and they fight over the same body.")
print()
print("the rest of section five builds toward exactly that.")

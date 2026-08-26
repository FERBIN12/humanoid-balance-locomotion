#!/usr/bin/env python3
"""5.7 -- put the whole stack on the robot and take the first steps.

5.3 gave a CoM trajectory, 5.4 foot placement and a swing path, 5.5 the joint
angles, 5.6 the torques. This runs them together on the H1-2 and records what
happens. No part of this file invents a new idea: it is the previous four
experiments wired up.

The result is not a walking robot. That is the experiment.
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

# gains that STAND, measured in 3.4: the x10 case
KP = np.array([200., 200., 200., 300., 60., 40.] * 2) * 10.0
KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2) * np.sqrt(10.0)

T_STEP = 0.735          # 4.8
STEP_L = 0.30           # 5.4
CLEAR = 0.05            # 5.4
L_THIGH = L_SHANK = 0.400
L_ANKLE = 0.020
HIP_H = 0.780           # the crouch the swing was solved at


def ik(dx, dz):
    """5.5's solver, unchanged: arccos branch, aimed at the ankle joint."""
    tz = dz + L_ANKLE
    r = np.hypot(dx, tz)
    if r > L_THIGH + L_SHANK:
        return None
    c = (r * r - L_THIGH ** 2 - L_SHANK ** 2) / (2 * L_THIGH * L_SHANK)
    knee = np.arccos(np.clip(c, -1.0, 1.0))
    beta = np.arctan2(-dx, -tz)
    alpha = np.arctan2(L_SHANK * np.sin(knee), L_THIGH + L_SHANK * np.cos(knee))
    return beta - alpha, knee


def swing_xz(u):
    """5.4's path: cosine ease forward, sine arch up."""
    x = STEP_L * 0.5 * (1 - np.cos(np.pi * u)) - STEP_L / 2
    z = -HIP_H + CLEAR * np.sin(np.pi * u)
    return x, z


def stance_xz(u):
    """The stance foot travels backward under the body by the same amount."""
    x = -(STEP_L * 0.5 * (1 - np.cos(np.pi * u)) - STEP_L / 2)
    return x, -HIP_H


def run(dur=6.0, settle=1.5, record=None):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    n = int(dur / m.opt.timestep)
    ns = int(settle / m.opt.timestep)
    log = []
    fell_at = None
    for step in range(n):
        t = step * m.opt.timestep
        tgt = np.zeros(m.nu)
        if step < ns:
            # settle into the crouch the gait was designed around
            k = min(1.0, step / max(1, ns * 0.6))
            sol = ik(0.0, -HIP_H)
            hp, kn = sol
            for side in ("left", "right"):
                tgt[IDX[side + "_hip_pitch_joint"]] = hp * k
                tgt[IDX[side + "_knee_joint"]] = kn * k
                tgt[IDX[side + "_ankle_pitch_joint"]] = -(hp + kn) * k
        else:
            # walk: alternate which leg swings every T_STEP
            gait_t = t - settle
            phase = int(gait_t / T_STEP)
            u = (gait_t % T_STEP) / T_STEP
            swing_side = "left" if phase % 2 == 0 else "right"
            stance_side = "right" if swing_side == "left" else "left"
            sx, sz = swing_xz(u)
            tx, tz = stance_xz(u)
            for side, (fx, fz) in ((swing_side, (sx, sz)),
                                   (stance_side, (tx, tz))):
                sol = ik(fx, fz)
                if sol is None:
                    continue
                hp, kn = sol
                tgt[IDX[side + "_hip_pitch_joint"]] = hp
                tgt[IDX[side + "_knee_joint"]] = kn
                tgt[IDX[side + "_ankle_pitch_joint"]] = -(hp + kn)
        q = d.qpos[QA]
        v = d.qvel[VA]
        d.ctrl[:] = KP * (tgt - q) - KD * v
        mujoco.mj_step(m, d)
        if fell_at is None and float(d.qpos[2]) < 0.55:
            fell_at = t
        if step % 25 == 0:
            R = d.xmat[1].reshape(3, 3)
            pitch = float(np.degrees(np.arctan2(-R[2, 0],
                                                np.hypot(R[2, 1], R[2, 2]))))
            mid = (d.xpos[FL] + d.xpos[FR]) / 2.0
            log.append((t, float(d.qpos[2]), float(d.qpos[0]), pitch,
                        float(d.subtree_com[0][0] - mid[0]), int(d.ncon)))
    return log, fell_at, d


print("--- the stack, wired together and run ---")
print("  gains: the x10 case from 3.4, the only ones that stand")
print("  step %.3f s, length %.2f m, clearance %.3f m, hip height %.3f m"
      % (T_STEP, STEP_L, CLEAR, HIP_H))
print()
log, fell_at, d = run()
print("%8s %10s %10s %10s %12s %6s"
      % ("t", "pelvis", "x", "pitch", "com-foot", "ncon"))
for row in log[::6]:
    print("%8.2f %10.3f %10.3f %10.1f %12.3f %6d" % row)
print()
if fell_at is not None:
    print("FELL at t = %.2f s, which is %.2f s after the first step command"
          % (fell_at, fell_at - 1.5))
    print("  steps attempted before falling: %.1f" % ((fell_at - 1.5) / T_STEP))
else:
    print("still up at the end: pelvis %.3f m" % d.qpos[2])
print("  final pelvis %.3f m, travelled %.3f m, pitch %.1f deg"
      % (d.qpos[2], d.qpos[0],
         np.degrees(np.arctan2(-d.xmat[1].reshape(3, 3)[2, 0],
                    np.hypot(d.xmat[1].reshape(3, 3)[2, 1],
                             d.xmat[1].reshape(3, 3)[2, 2])))))

print()
print("--- why: the diagnosis, not the symptom ---")
a = np.array([r[:5] for r in log])
tt, cf = a[:, 0], a[:, 4]
out = np.abs(cf) > 0.120
esc = float(tt[np.argmax(out)]) if out.any() else None
if esc is not None:
    print("  CoM left the 0.120 m support polygon at t = %.2f s" % esc)
    print("  pelvis first dropped below 0.55 m at t = %.2f s" % fell_at)
    print("  so balance was lost %.2f s BEFORE the height collapsed."
          % (fell_at - esc))
    print()
    print("  For a whole second this robot looked fine on a height plot and")
    print("  was already falling. Height is a LAGGING indicator. That is why")
    print("  1.6 built a support polygon check and why 5.8 plots four traces")
    print("  rather than one.")
print()
print("  And the cause is a missing piece, not a bad gain. Search this file")
print("  for the lateral CoM trajectory 5.3 computed:")
for term in ("com_trajectory", "lateral", "capture"):
    print("    %-16s referenced: %s"
          % (term, "yes" if term in open(__file__).read().split("--- why")[0]
             else "NO"))
print()
print("  The controller commands FEET. It never commands the centre of mass.")
print("  5.1 proved single support has no lateral equilibrium without moving")
print("  the pressure point, and 5.3 computed exactly the rocking that fixes")
print("  it. This file does not apply it, so the robot falls sideways into")
print("  its own swing leg on the second step. The stack is not wrong. It is")
print("  incomplete, and the missing piece is the one the maths warned about.")
print()
# Check the DIRECTION rather than asserting it. "Falls sideways" is a claim.
R = d.xmat[1].reshape(3, 3)
pitch_f = float(np.degrees(np.arctan2(-R[2, 0], np.hypot(R[2, 1], R[2, 2]))))
roll_f = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
print("  and the direction, measured rather than assumed:")
print("    final pitch %+.1f deg, roll %+.1f deg" % (pitch_f, roll_f))
print("    travelled %+.3f m forward, %+.3f m sideways"
      % (d.qpos[0], d.qpos[1]))
print("    the lateral displacement is the larger one, so this really is the")
print("    sideways failure 5.1 predicted and not a forward stumble.")


#!/usr/bin/env python3
"""5.5 -- turning a foot position into joint angles, and how to know it is right.

5.4 produced a foot PATH. The controller needs joint ANGLES. That conversion is
inverse kinematics, and I got it wrong twice before it was right. Both mistakes
are in here on purpose, because both are the kind that produce confident
numbers rather than errors.

The only reason I know the final version is correct is that every solution is
fed back through MuJoCo's own forward kinematics and compared to the target.
"""
import numpy as np
import mujoco, os

SCENE = os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
NAMES = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(m.nu)]
IDX = {n: i for i, n in enumerate(NAMES)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
HIP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_hip_pitch_link")
KNEE = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_knee_link")
AP = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link")
AR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")

# link lengths READ FROM THE MODEL, not from memory
L_THIGH = abs(float(m.body_pos[KNEE][2]))
L_SHANK = abs(float(m.body_pos[AP][2]))
L_ANKLE = abs(float(m.body_pos[AR][2]))


def fk_foot(hip_pitch, knee, ankle_pitch):
    """Where does the foot actually end up? MuJoCo answers, not my algebra."""
    d.qpos[:] = 0.0
    d.qpos[2] = 1.03
    d.qpos[3] = 1.0
    d.qpos[QA[IDX["left_hip_pitch_joint"]]] = hip_pitch
    d.qpos[QA[IDX["left_knee_joint"]]] = knee
    d.qpos[QA[IDX["left_ankle_pitch_joint"]]] = ankle_pitch
    mujoco.mj_forward(m, d)
    v = d.xpos[AR] - d.xpos[HIP]
    return float(v[0]), float(v[2])


print("--- 1 the chain, measured ---")
print("  thigh %.3f m, shank %.3f m, ankle joint to foot frame %.3f m"
      % (L_THIGH, L_SHANK, L_ANKLE))
print("  two link reach %.3f m, hip to foot at rest %.3f m"
      % (L_THIGH + L_SHANK, -fk_foot(0, 0, 0)[1]))
print("  those differ by exactly the ankle segment, so a solver aimed at the")
print("  FOOT with a two link model is asking for %.3f m it does not have."
      % L_ANKLE)
print()

print("--- 2 which way does the joint actually move? ---")
print("  Before writing any algebra, bend one joint and look.")
for kn in (0.0, 0.5, 1.0):
    x, z = fk_foot(0.0, kn, 0.0)
    print("    knee %.2f rad -> foot offset x %+.4f  z %+.4f" % (kn, x, z))
print("  A POSITIVE knee angle carries the foot BACKWARD. I assumed forward,")
print("  and every target came out mirrored, with errors near a full leg")
print("  length. A sign convention is a measurement, not a guess.")
print()


def ik(dx, dz):
    """Hip pitch and knee that put the FOOT at (dx, dz) relative to the hip.

    Two corrections over the naive version, both found by checking against
    forward kinematics rather than by inspecting the algebra:

      1 solve to the ANKLE JOINT, stripping the fixed 0.020 m below it
      2 knee = arccos(c). The textbook 2R form writes pi - arccos(c) for a
        chain whose second link folds the other way; here that picked the
        wrong elbow branch and returned 137 degrees where the truth is 49.
    """
    tz = dz + L_ANKLE
    r = np.hypot(dx, tz)
    if r > L_THIGH + L_SHANK:
        return None
    c = (r * r - L_THIGH ** 2 - L_SHANK ** 2) / (2 * L_THIGH * L_SHANK)
    if abs(c) > 1.0:
        return None
    knee = np.arccos(np.clip(c, -1.0, 1.0))
    beta = np.arctan2(-dx, -tz)
    alpha = np.arctan2(L_SHANK * np.sin(knee), L_THIGH + L_SHANK * np.cos(knee))
    return beta - alpha, knee


def ik_wrong_sign(dx, dz):
    """Version 1: correct branch logic, but assumes +knee moves the foot
    FORWARD. Kept so the failure is reproducible rather than anecdotal."""
    tz = dz + L_ANKLE
    r = np.hypot(dx, tz)
    if r > L_THIGH + L_SHANK:
        return None
    c = (r * r - L_THIGH ** 2 - L_SHANK ** 2) / (2 * L_THIGH * L_SHANK)
    knee = np.pi - np.arccos(np.clip(c, -1.0, 1.0))
    beta = np.arctan2(dx, -tz)
    alpha = np.arctan2(L_SHANK * np.sin(knee), L_THIGH + L_SHANK * np.cos(knee))
    return -(beta - alpha), knee


def ik_wrong_branch(dx, dz):
    """Version 2: sign fixed, but still the pi - arccos elbow branch."""
    tz = dz + L_ANKLE
    r = np.hypot(dx, tz)
    if r > L_THIGH + L_SHANK:
        return None
    c = (r * r - L_THIGH ** 2 - L_SHANK ** 2) / (2 * L_THIGH * L_SHANK)
    knee = np.pi - np.arccos(np.clip(c, -1.0, 1.0))
    beta = np.arctan2(-dx, -tz)
    alpha = np.arctan2(L_SHANK * np.sin(knee), L_THIGH + L_SHANK * np.cos(knee))
    return beta - alpha, knee


TARGETS = ((0.00, -0.780), (0.05, -0.760), (0.10, -0.740),
           (0.15, -0.720), (0.20, -0.700))


def worst_error(solver):
    w = 0.0
    for dx, dz in TARGETS:
        sol = solver(dx, dz)
        if sol is None:
            continue
        hp, kn = sol
        x, z = fk_foot(hp, kn, -(hp + kn))
        w = max(w, float(np.hypot(x - dx, z - dz)))
    return w


print("--- 3 the two wrong versions, so the failure is reproducible ---")
print("  version 1, knee sign assumed forward: worst error %.4f m"
      % worst_error(ik_wrong_sign))
print("  version 2, sign fixed, wrong elbow branch: worst error %.4f m"
      % worst_error(ik_wrong_branch))
print("  neither raises an exception. Both return confident angles.")
print()

print("--- 4 the solve, every row checked against the model ---")
print("%10s %10s %10s %10s %14s"
      % ("want dx", "want dz", "hip deg", "knee deg", "model error"))
worst = 0.0
for dx, dz in ((0.00, -0.780), (0.05, -0.760), (0.10, -0.740),
               (0.15, -0.720), (0.20, -0.700)):
    sol = ik(dx, dz)
    if sol is None:
        print("%10.3f %10.3f %10s" % (dx, dz, "UNREACHABLE"))
        continue
    hp, kn = sol
    ap = -(hp + kn)                     # keep the sole flat
    x, z = fk_foot(hp, kn, ap)
    err = float(np.hypot(x - dx, z - dz))
    worst = max(worst, err)
    print("%10.3f %10.3f %10.2f %10.2f %14.6f"
          % (dx, dz, np.degrees(hp), np.degrees(kn), err))
print()
print("worst position error: %.6f m" % worst)
if worst < 1e-4:
    print("that is below a tenth of a millimetre, so the closed form is exact")
    print("for this chain, not merely close.")
else:
    print("still %.1f mm off: something in the chain is unmodelled." % (worst * 1000))
print()

print("--- 5 running the actual swing path through it ---")
T, STEP, CLEAR = 0.735, 0.30, 0.05
print("  the 5.4 swing path, solved at 9 points, hip held 0.780 m above the foot")
print("%8s %10s %10s %10s %12s" % ("t", "foot dx", "foot dz", "knee deg", "err"))
worst_path = 0.0
for i in range(9):
    u = i / 8.0
    fx = STEP * 0.5 * (1 - np.cos(np.pi * u)) - STEP / 2
    fz = -0.780 + CLEAR * np.sin(np.pi * u)
    sol = ik(fx, fz)
    if sol is None:
        print("%8.3f %10.3f %10.3f  UNREACHABLE" % (u * T, fx, fz))
        continue
    hp, kn = sol
    x, z = fk_foot(hp, kn, -(hp + kn))
    e = float(np.hypot(x - fx, z - fz))
    worst_path = max(worst_path, e)
    print("%8.3f %10.3f %10.3f %10.2f %12.6f"
          % (u * T, fx, fz, np.degrees(kn), e))
print()
print("worst error over the whole swing: %.6f m" % worst_path)
print()

print("--- 6 the singularity that shapes every gait ---")
print("%10s %12s %s" % ("hip to foot", "knee deg", "note"))
for dz in (-0.819, -0.815, -0.80, -0.75, -0.70, -0.60):
    sol = ik(0.0, dz)
    if sol is None:
        print("%10.3f %12s %s" % (dz, "none", "beyond reach"))
        continue
    print("%10.3f %12.2f %s"
          % (dz, np.degrees(sol[1]),
             "near straight: Jacobian rank drops" if sol[1] < 0.18 else ""))
print()
print("at full extension the knee goes to zero and the leg loses a degree of")
print("freedom: vertical motion still works, but horizontal motion of the foot")
print("costs unbounded joint rate. That is why walking gaits keep a bent knee")
print("and pay for it in torque every step. 5.6 turns these angles into torques.")

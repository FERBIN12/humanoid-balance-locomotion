#!/usr/bin/env python3
"""The hip strategy, and why this robot does not have one.

The textbook second line of defence, after the ankle, is the hip strategy:
pitch the torso sharply to generate a reaction that shifts the centre of
pressure. I planned a whole experiment on it. Then I looked at the model.

H1-2 has ONE waist joint, torso_joint, and its axis is [0, 0, 1]. That is yaw.
There is no waist pitch anywhere in this robot: the torso is rigid with respect
to the pelvis in the sagittal plane. So the classical hip strategy is not
available, and pretending otherwise would be teaching a fiction.

What IS available is measured below.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
G, H = 9.81, 0.937
OMEGA = np.sqrt(G / H)
MASS = float(m.body_mass.sum())
FOOT_L = 0.240
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
idx = {n: i for i, n in enumerate(names)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}


def jnt_of(act):
    return m.actuator_trnid[act][0]


# --- 1 the audit that killed the plan ---------------------------------------
print("every joint that could plausibly pitch the upper body:")
for n in ("torso_joint", "left_hip_pitch_joint", "left_shoulder_pitch_joint"):
    if n.replace("_joint", "") + "_joint" in idx or n in idx:
        j = jnt_of(idx[n])
        print("  %-30s axis %s   range %s"
              % (n, np.round(m.jnt_axis[j], 2), np.round(m.jnt_range[j], 2)))
print()
print("torso_joint axis is [0 0 1]: YAW. There is no waist pitch on this robot.")
print("so the classical hip strategy has nothing to act on.")
print()

# --- 2 what leverage each joint really has ----------------------------------
# Measure it: perturb one joint, see how far the CoM moves relative to a foot.
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")


def leverage(act, span=0.30):
    got = []
    for a in (-span, span):
        d = mujoco.MjData(m)
        d.qpos[:3] = [0.0, 0.0, 1.03]
        d.qpos[QA[act]] = a
        mujoco.mj_forward(m, d)
        c = (m.body_mass[:, None] * d.xipos).sum(0) / MASS
        got.append(c - d.xpos[FL])
    return float((got[1] - got[0])[0] / (2 * span))


print("d(CoM x relative to the foot) / d(joint angle), measured, m/rad:")
CAND = ["torso_joint", "left_shoulder_pitch_joint", "left_hip_pitch_joint",
        "left_knee_joint", "left_ankle_pitch_joint"]
lev = {}
for n in CAND:
    lev[n] = leverage(idx[n])
    print("  %-30s %+8.4f" % (n, lev[n]))
print()
print("the torso contributes %.4f m/rad, which is nothing." % abs(lev["torso_joint"]))
print("all the sagittal leverage on this robot lives in the LEGS.")
print()

# --- 3 so what is the second line of defence? -------------------------------
# Two candidates remain: the arms (angular momentum) and the legs themselves
# (which is really just a bigger ankle strategy, plus stepping).
def torque_limit(n):
    return float(m.jnt_actfrcrange[jnt_of(idx[n])][1])


TAU_SH = torque_limit("left_shoulder_pitch_joint") * 2
print("the arms, which DO pitch:")
print("  shoulder pitch limit %.0f Nm per side, %.0f Nm together"
      % (TAU_SH / 2, TAU_SH))
cop_arm = TAU_SH / (MASS * G)
print("  CoP shift = tau/(m g) = %.4f m = %.2f foot half lengths"
      % (cop_arm, cop_arm / (FOOT_L / 2)))
print("  and 3.10 measured it lasting 0.242 s, which is %.2f falling constants"
      % (0.242 * OMEGA))
print()
print("that is the whole upper body contribution on this machine: about one")
print("extra foot, for about eight tenths of the time you have.")
print()
print("so the ladder on H1-2 is not ankle, hip, step. It is ankle, ARMS, step,")
print("and the middle rung is much weaker than the textbook one. Which makes")
print("stepping arrive sooner than it would on a robot with a waist.")
print()

# --- 4 the general lesson ---------------------------------------------------
print("the transferable point: a strategy is a property of the MECHANISM, not")
print("of the control literature. Check that the joint exists, check its axis,")
print("and check its leverage, before you build a controller on top of it.")

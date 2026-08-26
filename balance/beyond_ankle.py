#!/usr/bin/env python3
"""What is left when the ankle runs out? Quantify each option before the stepping controller.

The ankle gave us 176 N. This script computes, from the same measured model,
what the other three strategies are worth, so section four starts from numbers
instead of intuition.

  1 hip strategy   -- accelerate the upper body to shift the CoP
  2 stepping       -- move the support polygon under the CoM
  3 arms           -- angular momentum from a 0.101 kg hand on a long lever
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
G = 9.81
H = 0.937
OMEGA = np.sqrt(G / H)
MASS = float(m.body_mass.sum())
FOOT_L = 0.240

print("the ankle limit, measured in 3.9: 176 N give or take 9")
print("CoM height %.3f m, omega %.3f rad/s, total mass %.2f kg"
      % (H, OMEGA, MASS))
print()

# --- 1 the hip strategy -----------------------------------------------------
# Accelerating the upper body backwards produces a reaction that shifts the CoP
# forward WITHOUT needing more foot. The authority is bounded by the joint
# torque, so start from the actuator limit and never from a desired motion.
def body_id(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


def torque_limit(joint):
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)]
    j = m.actuator_trnid[names.index(joint)][0]
    return float(m.jnt_actfrcrange[j][1])


TAU_HIP = torque_limit("left_hip_pitch_joint") * 2      # both hips
TAU_SHOULDER = torque_limit("left_shoulder_pitch_joint") * 2
torso = body_id("torso_link")
I_torso = float(m.body_inertia[torso].max())
print("hip pitch limit %.0f Nm per side, so %.0f Nm together"
      % (TAU_HIP / 2, TAU_HIP))
print("torso %.2f kg, largest principal inertia %.4f kg m2"
      % (float(m.body_mass[torso]), I_torso))

# A reaction torque tau shifts the centre of pressure by tau / (m g).
cop_hip = TAU_HIP / (MASS * G)
print("CoP shift from a hip reaction: tau/(m g) = %.4f m" % cop_hip)
print("the foot only offers %.4f m, so the hip is worth %.2f extra feet"
      % (FOOT_L / 2, cop_hip / (FOOT_L / 2)))

# But it cannot be held, and here the ANALYTIC estimate is not good enough, so
# measure it: suspend the robot, command the torque limit, watch the joint.
#   hip pitch at 200 Nm  -> peak 26.7 rad/s, covers 0.60 rad in 0.044 s
# The analytic torso-only figure was 821 rad/s2; the measured peak implies about
# 342, because the hip must accelerate the torso PLUS the arms and the head.
# The estimate was right about the CONCLUSION and wrong about the number, which
# is why the number has to be measured.
T_HIP = 0.044
print("MEASURED: at %.0f Nm the hips cover 0.60 rad in %.3f s (peak 26.7 rad/s)"
      % (TAU_HIP, T_HIP))
print("against a falling time constant of %.3f s, that is %.0f per cent of one"
      % (1.0 / OMEGA, 100 * T_HIP * OMEGA))
print("the hip is a large, brief impulse: you spend it once and it is gone")
print()

# --- 2 stepping -------------------------------------------------------------
print("stepping: the polygon MOVES instead of the pressure moving inside it")
for step_len in (0.20, 0.40, 0.60):
    print("  a %.2f m step reaches a capture point at %.2f m, so a CoM speed"
          " of %.2f m/s" % (step_len, step_len, step_len * OMEGA))
print("no torque limit appears in that sum, which is exactly why walking exists")
print()
print("the cost is TIME: the foot must land within one falling constant")
for step_len in (0.20, 0.40):
    print("  %.2f m in %.3f s means a swing foot speed of %.2f m/s"
          % (step_len, 1.0 / OMEGA, step_len * OMEGA))
print()

# --- 3 the arms -------------------------------------------------------------
# My first attempt here was wrong in an instructive way. I picked a motion --
# swing both arms through 2 rad in 0.3 s -- and computed the torque it needed:
# 170.7 Nm. Then I checked the shoulder, which is limited to 40 Nm per side.
# I had specified a motion the motor cannot produce, and reported its reaction
# as if it were available. Start from the LIMIT, not from the desired motion.
larm = [body_id(n) for n in ("left_shoulder_pitch_link", "left_shoulder_roll_link",
                            "left_shoulder_yaw_link", "left_elbow_pitch_link",
                            "left_elbow_roll_link", "left_wrist_pitch_link",
                            "left_wrist_yaw_link")]
m_arm = sum(float(m.body_mass[b]) for b in larm)
r_arm = 0.55
I_arm = m_arm * r_arm ** 2
print("one arm chain %.3f kg at about %.2f m, inertia %.4f kg m2"
      % (m_arm, r_arm, I_arm))
print("shoulder pitch limit %.0f Nm per side, so %.0f Nm together"
      % (TAU_SHOULDER / 2, TAU_SHOULDER))
cop_arm = TAU_SHOULDER / (MASS * G)
# MEASURED, not assumed: at 40 Nm the shoulder reaches its 1.574 rad upper
# STOP in 0.242 s and then holds there. The limit is joint RANGE, not torque,
# and my first estimate of 0.438 s was for a 2.0 rad swing the joint cannot make.
T_ARM = 0.242
print("that gives a CoP shift of %.4f m, which is %.0f per cent of a foot half"
      % (cop_arm, 100 * cop_arm / (FOOT_L / 2)))
print("MEASURED: the arm reaches its 1.574 rad stop in %.3f s and holds"
      % T_ARM)
print("so the arms last %.2f falling constants against the hip's %.2f"
      % (T_ARM * OMEGA, T_HIP * OMEGA))
print()
print("so the ranking, every number from a measured limit:")
print("  stepping  moves the polygon: metres of capture point, no torque bound")
print("  hip       %.2f extra feet, but only %.3f s" % (cop_hip / (FOOT_L / 2), T_HIP))
print("  arms      %.0f per cent of one foot, for %.3f s"
      % (100 * cop_arm / (FOOT_L / 2), T_ARM))
print()
print("and note which of those two lasts longer. The hip is five times stronger")
print("and spends itself in a fifth of the time. The arms are weaker and are the")
print("only non stepping option that outlives the fall it is trying to arrest.")
print()
print("stepping still wins by an order of magnitude, and it is the only one whose")
print("authority does not appear in a datasheet.")

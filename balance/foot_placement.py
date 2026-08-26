#!/usr/bin/env python3
"""Where should the foot LAND? The capture point tells you, with caveats.

3.7 derived the capture point and 3.8 verified it as a boundary for an ankle
only controller. 4.3 then measured recoveries at 2.81x and 3.34x that bound,
because arms add a mechanism the criterion never modelled.

Now we use the capture point differently: not as a go/no-go test, but as a
TARGET. Put the foot at the capture point and the pendulum comes to rest.
This script works out what that actually demands.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
G, H = 9.81, 0.937
OMEGA = np.sqrt(G / H)
FOOT_L, FOOT_W = 0.240, 0.110
STANCE_W = 0.326
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
idx = {n: i for i, n in enumerate(names)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}

print("omega %.3f rad/s, so 1/omega = %.3f s" % (OMEGA, 1.0 / OMEGA))
print()

# --- 1 the target is the capture point, not the CoM -------------------------
print("if you step so the new support is AT the capture point, the pendulum")
print("comes to rest. Stepping to the CoM is not enough: the CoM is still moving.")
print()
print("%10s %10s %14s %14s" % ("lean", "speed", "step to CoM", "step to capture"))
for lean, v in ((0.05, 0.30), (0.08, 0.50), (0.10, 0.80), (0.12, 1.10)):
    print("%10.3f %10.2f %14.3f %14.3f"
          % (lean, v, lean, lean + v / OMEGA))
print()
print("the difference is v/omega, which at 1.10 m/s is %.3f m of extra step."
      % (1.10 / OMEGA))
print("step to the CoM and you arrive under a mass that is still travelling.")
print()

# --- 2 how far CAN the foot reach? -----------------------------------------
# My first attempt modelled the leg as two 0.400 m links from a hip 0.10 m below
# the CoM. It said even a 0.20 m step was infeasible, and produced a nan. The
# error: the measured hip sits 0.820 m above the ankle while thigh plus shank is
# only 0.800 m, because the ankle frame is 0.047 m off the floor and the foot
# adds length below the roll joint. A two link model of a three segment leg.
#
# So measure it. Sweep hip, knee and ankle, keep only poses where the swing foot
# is AT landing height, and take the furthest forward.
Z_LAND = 0.047
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")


def max_step(pelvis_z):
    best = 0.0
    for hp in np.arange(0.0, 1.60, 0.03):
        for kn in np.arange(0.0, 2.05, 0.03):
            for an in np.arange(-0.85, 0.52, 0.06):
                d = mujoco.MjData(m)
                d.qpos[:3] = [0.0, 0.0, pelvis_z]
                d.qpos[QA[idx["left_hip_pitch_joint"]]] = -hp
                d.qpos[QA[idx["left_knee_joint"]]] = kn
                d.qpos[QA[idx["left_ankle_pitch_joint"]]] = an
                mujoco.mj_forward(m, d)
                if abs(float(d.xpos[FL][2]) - Z_LAND) < 0.015:
                    best = max(best, float(d.xpos[FL][0]))
    return best


print("how far can the swing foot reach and still touch the floor?")
print("%12s %14s %16s" % ("pelvis z", "max step", "CoM speed"))
rows = []
for pz in (1.03, 0.98, 0.93, 0.88, 0.83):
    s_len = max_step(pz)
    rows.append((pz, s_len))
    print("%12.2f %14.4f %16.2f" % (pz, s_len, s_len * OMEGA))
print()
tall, low = rows[0][1], rows[3][1]
print("at the standing height of 1.03 m the step is only %.3f m, because the" % tall)
print("swing leg must stay nearly straight to reach the ground and has no room")
print("left to swing forward.")
print("drop the pelvis to 0.88 m and the step becomes %.3f m: %.1fx further."
      % (low, low / tall))
print("recoverable CoM speed goes from %.2f to %.2f m/s."
      % (tall * OMEGA, low * OMEGA))
print()
print("that is why every walking humanoid you have ever seen walks with bent")
print("knees. It is not for shock absorption and it is not a stylistic choice.")
print("A straight legged robot cannot take a long step, because step length and")
print("hip height trade against each other through a fixed leg.")
print()

# --- 3 the deadline --------------------------------------------------------
print("the other constraint is TIME: the foot must arrive within about one")
print("falling constant, %.3f s." % (1.0 / OMEGA))
print()
print("%10s %14s %16s" % ("step", "foot speed", "hip rate needed"))
for s_len in (0.20, 0.40, 0.50):
    v_foot = s_len / (1.0 / OMEGA)
    print("%10.2f %14.2f %16.2f" % (s_len, v_foot, v_foot / 0.80))
print()
print("3.10 measured the hip reaching 26.7 rad/s at its torque limit, so those")
print("rates are comfortable. The swing leg is not the bottleneck. Geometry is.")
print()

# --- 4 what a step costs ---------------------------------------------------
print("and while one foot is airborne the support polygon is a SINGLE foot:")
print("  double support lateral half width  %.3f m" % (STANCE_W / 2 + FOOT_W / 2))
print("  single support lateral half width  %.3f m" % (FOOT_W / 2))
print("  so lifting a foot costs %.1fx of your lateral authority"
      % ((STANCE_W / 2 + FOOT_W / 2) / (FOOT_W / 2)))
print()
print("that is the real price of a step, and it is why a humanoid falling")
print("sideways is in far more trouble than one falling forwards.")

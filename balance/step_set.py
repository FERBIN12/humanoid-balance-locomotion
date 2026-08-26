#!/usr/bin/env python3
"""Which steps are reachable, and when does no step exist?

4.4 measured the max forward step as a function of pelvis height. That is one
number per height. The real question a controller asks is different: given where
my mass is going RIGHT NOW, is there a foot placement that stops it?

Three things bound the answer:
  1 GEOMETRY  -- the swing foot has to reach the target and touch the floor
  2 TIME      -- it has to arrive within about one falling constant
  3 KINEMATICS of the OTHER leg -- the stance leg must still support the pelvis

When the required placement is outside all three, no step exists, and the robot
is going down no matter what it does. That set is worth knowing exactly.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
G, H = 9.81, 0.937
OMEGA = np.sqrt(G / H)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
idx = {n: i for i, n in enumerate(names)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
Z_LAND = 0.047
FOOT_W, STANCE_W = 0.110, 0.326


def reach(pelvis_z, coarse=0.04):
    """Furthest the swing foot can land, measured, in three directions.

    Two traps here, both of which gave me a wrong answer first:

    1 report the DISPLACEMENT from the neutral foot position, not the absolute
      y. The left foot already sits at y = +0.163, so reading xpos directly
      counts the existing stance width as step reach and reported 0.653 m.

    2 outward and inward are not the same problem. Crossing the foot over runs
      into the other leg, so any pose whose foot separation drops below one
      foot width is not a step, it is a collision.
    """
    d0 = mujoco.MjData(m)
    d0.qpos[:3] = [0.0, 0.0, pelvis_z]
    mujoco.mj_forward(m, d0)
    y0 = float(d0.xpos[FL][1])

    fwd = 0.0
    for hp in np.arange(0.0, 1.60, coarse):
        for kn in np.arange(0.0, 2.05, coarse):
            for an in np.arange(-0.85, 0.52, 0.08):
                d = mujoco.MjData(m)
                d.qpos[:3] = [0.0, 0.0, pelvis_z]
                d.qpos[QA[idx["left_hip_pitch_joint"]]] = -hp
                d.qpos[QA[idx["left_knee_joint"]]] = kn
                d.qpos[QA[idx["left_ankle_pitch_joint"]]] = an
                mujoco.mj_forward(m, d)
                if abs(float(d.xpos[FL][2]) - Z_LAND) < 0.015:
                    fwd = max(fwd, float(d.xpos[FL][0]))

    out = inn = 0.0
    for hr in np.arange(-0.43, 3.10, 0.03):
        for kn in np.arange(0.0, 1.30, 0.05):
            d = mujoco.MjData(m)
            d.qpos[:3] = [0.0, 0.0, pelvis_z]
            d.qpos[QA[idx["left_hip_roll_joint"]]] = hr
            d.qpos[QA[idx["left_knee_joint"]]] = kn
            mujoco.mj_forward(m, d)
            if abs(float(d.xpos[FL][2]) - Z_LAND) >= 0.020:
                continue
            sep = abs(float(d.xpos[FL][1]) - float(d.xpos[FR][1]))
            if sep <= FOOT_W:
                continue                     # the feet would collide
            disp = float(d.xpos[FL][1]) - y0
            if disp > 0:
                out = max(out, disp)
            else:
                inn = max(inn, -disp)
    return fwd, out, inn


print("the reachable step set, measured as DISPLACEMENT from the neutral foot")
print()
print("%10s %11s %11s %11s" % ("pelvis z", "forward", "outward", "inward"))
for pz in (1.03, 0.93, 0.88):
    f, o, i2 = reach(pz)
    print("%10.2f %11.4f %11.4f %11.4f" % (pz, f, o, i2))
print()
F, OUT, INN = reach(0.88)
print("at the 0.88 m crouch: %.3f forward, %.3f outward, %.3f inward."
      % (F, OUT, INN))
print("outward is %.1fx the inward reach, because crossing the foot over runs"
      % (OUT / INN))
print("into the other leg. The step set is not symmetric, and the direction it")
print("is thin in is the one we are already weakest in.")
print()

# --- when does no step exist? ---------------------------------------------
print("so: given a lean and a speed, is there a step? The capture point has to")
print("land inside that set.")
print()
print("%8s %8s %11s %14s" % ("lean", "speed", "capture", "step exists"))
for lean, v in ((0.05, 0.5), (0.08, 1.0), (0.10, 1.4), (0.12, 1.8), (0.15, 2.2)):
    cap = lean + v / OMEGA
    print("%8.3f %8.2f %11.3f %14s" % (lean, v, cap, "yes" if cap <= F else "NO"))
print()
v_fwd = (F - 0.10) * OMEGA
print("from a 0.10 m lean the fastest recoverable forward speed is %.2f m/s."
      % v_fwd)
print("past that there is no foot placement that stops the fall. That is not a")
print("control failure, it is a kinematic one, and no amount of tuning fixes it.")
print()

# --- and sideways -----------------------------------------------------------
print("now the same sum sideways, in the hard direction:")
v_in = INN * OMEGA
v_out = OUT * OMEGA
for v in (0.3, 0.5, 0.7, 0.9):
    cap = v / OMEGA
    print("  %.1f m/s -> capture at %.3f m: outward %s, inward %s"
          % (v, cap, "yes" if cap <= OUT else "NO",
             "yes" if cap <= INN else "NO"))
print()
print("so stepping catches a %.2f m/s drift toward the swing foot and only"
      % v_out)
print("%.2f m/s away from it. Falling to your left, you step left, easy." % v_in)
print("Falling to your right, the left foot has to cross over, and that is")
print("where the number more than halves.")
print()
print("forward we tolerate %.2f m/s. So the worst direction on this robot is"
      % v_fwd)
print("a sideways push toward the STANCE foot, at %.2f m/s, which is %.0f per"
      % (v_in, 100 * v_in / v_fwd))
print("cent of the forward figure. Lateral push tests are the hard ones, and")
print("this is the number that says why.")

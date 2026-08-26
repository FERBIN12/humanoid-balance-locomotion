#!/usr/bin/env python3
"""5.4 -- where to put the foot, and how to get it there.

5.3 produced the CoM trajectory. This produces the OTHER half of a walking
pattern: the swing foot's path through the air. Two separate questions:

  1 WHERE does the foot land?    -> capture point, not the CoM
  2 HOW does it get there?       -> a trajectory that respects the clock

Every constant is measured elsewhere in this project. Nothing invented.
"""
import numpy as np
import mujoco, os

G = 9.81
H = 0.937                     # CoM height standing (1.1)
OMEGA = np.sqrt(G / H)
TAU = 1.0 / OMEGA
T_STEP = 0.735                # single support budget (4.8)
FOOT_L, FOOT_W = 0.240, 0.110
STANCE_W = 0.326
STEP_MAX = 0.493              # reachable forward step (4.5)
DT = 0.002

print("constants, all measured earlier in this project:")
print("  omega %.4f rad/s, tau %.4f s" % (OMEGA, TAU))
print("  single support %.3f s (4.8), reachable step %.3f m (4.5)"
      % (T_STEP, STEP_MAX))
print()

# --- 1 where to land: the capture point ------------------------------------
print("--- 1 WHERE: the capture point, not the centre of mass ---")
# Compare LIKE WITH LIKE. Both columns are offsets from where the CoM is NOW:
#   under the CoM  -> 0 by definition, you step to where the mass already is
#   capture point  -> v/omega, the offset that brings the CoM to rest
# My first version tabulated v*T_STEP, which is how far the CoM TRAVELS during
# the step, and then claimed stepping under the CoM "undershoots". That
# compared two different reference points and got the direction wrong.
print("both columns are offsets from where the CoM is right now:")
print("%10s %14s %14s %12s"
      % ("com vel", "under the CoM", "capture pt", "reachable"))
for v in (0.0, 0.3, 0.6, 0.9, 1.2, 1.5):
    cap = v / OMEGA
    print("%8.1f m/s %12.3f m %12.3f m %12s"
          % (v, 0.0, cap, "yes" if cap <= STEP_MAX else "NO"))
print()
print("stepping under the centre of mass asks for a zero offset at every")
print("speed, which is only correct at standstill. The offset you actually")
print("need is v/omega, and at 1.0 m/s that is %.3f m: a third of a step."
      % (1.0 / OMEGA))
print("Step under the mass while moving and you land BEHIND the point that")
print("would have stopped you, so the CoM keeps going and you must step again.")
print()
# The reach bounds the speed, but ONLY once you say what lean it is spending
# on. 4.5 measured 1.27 m/s from a 0.10 m lean, because the capture point is
# lean + v/omega and the lean eats reach before the velocity term gets any.
# Quoting the zero-lean number alone contradicts 4.5 for no reason.
print("the reach bounds the speed, and the bound depends on the lean:")
print("%10s %14s %14s" % ("lean", "reach left", "max speed"))
for lean in (0.0, 0.05, 0.10, 0.15):
    left = STEP_MAX - lean
    print("%8.3f m %12.3f m %10.2f m/s" % (lean, left, left * OMEGA))
print()
print("from upright, %.2f m/s. From the 0.10 m lean 4.5 used, %.2f m/s, which"
      % (STEP_MAX * OMEGA, (STEP_MAX - 0.10) * OMEGA))
print("is the 1.27 m/s that experiment measured. Same formula, different lean.")
print("Either way it is a kinematic ceiling, not a control failure.")
print()

# --- 2 how to get there: the swing trajectory -------------------------------
print("--- 2 HOW: the swing path, and what bounds its shape ---")


def swing(t, T=T_STEP, L=0.30, clearance=0.05):
    """Foot position over one swing. Horizontal: a cosine ease so velocity is
    zero at BOTH ends (a foot that lands moving scuffs). Vertical: a single
    sine arch, zero at both ends, peak at mid swing."""
    u = np.clip(t / T, 0.0, 1.0)
    x = L * 0.5 * (1.0 - np.cos(np.pi * u))
    z = clearance * np.sin(np.pi * u)
    return x, z


def swing_vel(t, T=T_STEP, L=0.30, clearance=0.05):
    u = np.clip(t / T, 0.0, 1.0)
    xd = L * 0.5 * np.pi / T * np.sin(np.pi * u)
    zd = clearance * np.pi / T * np.cos(np.pi * u)
    return xd, zd


n = int(round(T_STEP / DT))
t = np.arange(n + 1) * DT
X, Z = swing(t)
XD, ZD = swing_vel(t)

print("a 0.300 m step in %.3f s with 0.050 m clearance:" % T_STEP)
print("  peak horizontal speed %.3f m/s at mid swing" % XD.max())
print("  touchdown horizontal speed %.4f m/s" % abs(XD[-1]))
print("  touchdown vertical speed %.4f m/s" % abs(ZD[-1]))
print("  lift off vertical speed %.3f m/s" % ZD[0])
print()
if abs(XD[-1]) < 1e-9:
    print("  horizontal touchdown speed is EXACTLY zero, which is the point of")
    print("  the cosine: a foot that lands moving forward scuffs or trips.")
print("  the vertical speed at touchdown is %.3f m/s, and that is NOT zero."
      % abs(ZD[-1]))
print("  A sine arch lands the foot moving DOWN at its steepest. That is a")
print("  real defect of this shape and 5.7 is where it bites.")
print()

# --- 3 the clearance the terrain actually demands ---------------------------
print("--- 3 how much clearance, and what it costs ---")
print("%12s %14s %16s" % ("clearance", "peak knee flex", "peak lift speed"))
for c in (0.02, 0.05, 0.10, 0.15):
    _, zd = swing_vel(t, clearance=c)
    # knee flexion needed to raise the foot by c with a 0.400+0.400 leg:
    # hip height fixed, so 2*0.4*cos(knee/2) = 0.800 - c
    arg = np.clip((0.800 - c) / 0.800, -1.0, 1.0)
    knee = 2.0 * np.arccos(arg)
    print("%10.3f m %12.2f deg %14.3f m/s"
          % (c, np.degrees(knee), abs(zd).max()))
print()
print("clearance is cheap in knee angle and expensive in SPEED, because the")
print("clock is fixed: the same %.3f s has to cover more vertical travel."
      % T_STEP)
print()

# --- 4 does the model agree the pose is reachable? -------------------------
print("--- 4 asking the model, not my arithmetic ---")
SCENE = os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(m.nu)]
idx = {n_: i for i, n_ in enumerate(names)}
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
mujoco.mj_forward(m, d)
z0 = float(d.xpos[FL][2])
print("  left foot at spawn: z = %.4f m" % z0)
worst = 0.0
for c in (0.02, 0.05, 0.10):
    arg = np.clip((0.800 - c) / 0.800, -1.0, 1.0)
    knee = 2.0 * np.arccos(arg)
    d.qpos[QA[idx["left_hip_pitch_joint"]]] = -knee / 2
    d.qpos[QA[idx["left_knee_joint"]]] = knee
    d.qpos[QA[idx["left_ankle_pitch_joint"]]] = -knee / 2
    mujoco.mj_forward(m, d)
    got = float(d.xpos[FL][2]) - z0
    err = abs(got - c)
    worst = max(worst, err)
    print("  asked %.3f m of lift, model gives %.4f m  (error %.4f)"
          % (c, got, err))
print()
if worst < 0.006:
    print("  the two link arithmetic agrees with the model to %.1f mm, so the"
          % (worst * 1000))
    print("  clearance table above is trustworthy for THIS pose family.")
else:
    print("  the arithmetic and the model disagree by %.1f mm: trust the model."
          % (worst * 1000))
print()
print("that is the swing half of the pattern. 5.5 turns these foot positions")
print("into joint angles, which is where the three segment leg bites again.")

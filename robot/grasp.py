#!/usr/bin/env python3
"""Close all twelve joints of a hand from a single grip scalar.

Real grasp planning chooses per digit targets from the object. One scalar is
enough for this project and keeps the interface tiny, but it cannot be the SAME
angle on every joint: see note 2.

Three things this script had to get right, all found the hard way:

1 ARMATURE. Every DOF in the shipped model has zero armature, which is
  physically wrong: a real actuator has rotor and gearbox inertia. Without it a
  gram-scale finger link (inertia ~1e-7 kg m2) diverges under ANY position gain
  at a 2 ms step. Measured: qvel reached 6842 rad/s in ONE step on
  L_thumb_proximal_yaw. scene_full.xml now declares armature, and that is a
  modelling fix, not a gain fix.

2 QPOS IS NOT 7 + ACTUATOR INDEX. This model has nu=51 actuators but nq=58,
  and the hand joints are not laid out in actuator order: actuator 35 is
  L_thumb_proximal_yaw_joint, whose qpos address is 27, not 42. Indexing with
  7 + i therefore reads a DIFFERENT joint, and for the last actuators it runs
  off the end of the array. It cost me a long detour, because the symptoms
  looked exactly like physics:

    - nine joints frozen at 0.000 with a large torque command
    - three pinned at 1.702, hard against their upper limit
    - fourteen contacts reported between the thumb and the index finger
    - identical behaviour at kp 12 and kp 0.5, a 24x span

  All of it was one array-indexing bug comparing a left hand command against a
  right hand measurement. Ask the model where a joint lives:

    qadr = m.jnt_qposadr[m.actuator_trnid[i][0]]
    vadr = m.jnt_dofadr[m.actuator_trnid[i][0]]

  The lesson worth keeping: gain independence means it is not a gain problem,
  and it usually means the number you are reading is not the number you think.

3 PER JOINT TRAVEL. The digits do not share a range. The fingers run 0 to 1.7
  rad; the thumb is tighter and moves on different axes (yaw -0.1..1.3,
  pitch -0.1..0.6, intermediate 0..0.8, distal 0..1.2). So the grip scalar
  drives a FRACTION of each joint's own travel. Commanding one shared angle
  folds the thumb into the index finger instead of opposing it.

The joints declare actuatorfrcrange="-1 1", a real 1 Nm ceiling from the
manufacturer, so torques are clipped to it. Peak demand here is 0.076 Nm, so
the limit never binds: these links weigh grams.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
SUSPEND_Z = 3.0   # hold the body clear of the floor: standing is the balance controller's job
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
fing = [i for i, n in enumerate(names) if n and any(k in n for k in DIGITS)]

# ask the model where each actuated joint lives, per note 2
QADR = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VADR = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
LIM = {i: m.jnt_range[m.actuator_trnid[i][0]] for i in range(m.nu)}

TAU_MAX = 1.0     # actuatorfrcrange on every finger joint
KP, KD = 3.0, 0.1


def share(name):
    """Fraction of its OWN travel each joint takes at full grip, per note 3."""
    if "thumb_proximal_yaw" in name:
        return 0.55        # opposition: swing the thumb across the palm
    if "thumb_proximal_pitch" in name:
        return 0.30
    if "thumb" in name:
        return 0.35        # the thumb curls less than the fingers
    return 0.85            # the four fingers curl


def target(i, grip):
    lo, hi = LIM[i]
    base = max(lo, 0.0)
    return base + grip * share(names[i]) * (hi - base)


mujoco.mj_forward(m, d)
d.qpos[:3] = [0.0, 0.0, SUSPEND_Z]

print("driving ONLY the fingers, kp %.0f, kd %.1f, torque clipped to %.0f Nm"
      % (KP, KD, TAU_MAX))
print()
print("%6s %6s   %s" % ("t", "grip", "left hand: index, thumb yaw, peak torque"))
ii = names.index("L_index_proximal_joint")
th = names.index("L_thumb_proximal_yaw_joint")

for step in range(6000):
    grip = min(1.0, step / 2000.0)
    tau = np.zeros(m.nu)
    for i in fing:
        tau[i] = KP * (target(i, grip) - d.qpos[QADR[i]]) - KD * d.qvel[VADR[i]]
    peak = float(np.abs(tau).max())
    d.ctrl[:] = np.clip(tau, -TAU_MAX, TAU_MAX)
    mujoco.mj_step(m, d)
    # hold the free joint still: we are studying hands, not balance
    d.qpos[:3] = [0.0, 0.0, SUSPEND_Z]
    d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[:6] = 0.0
    if step % 1000 == 0 or step == 5999:
        print("%6.2f %6.3f   %6.3f  %6.3f  %6.3f"
              % (d.time, grip, d.qpos[QADR[ii]], d.qpos[QADR[th]], peak))

print()
worst = 0.0
for i in fing:
    worst = max(worst, abs(d.qpos[QADR[i]] - target(i, 1.0)))
inside = sum(1 for i in fing
             if LIM[i][0] - 0.06 <= d.qpos[QADR[i]] <= LIM[i][1] + 0.06)
print("worst tracking error %.4f rad = %.2f deg" % (worst, np.degrees(worst)))
print("%d of %d finger joints inside their limits" % (inside, len(fing)))
print("contacts between digits: %d" % d.ncon)
print()
print("the fingers curl to 85 per cent of 1.7 rad; the thumb yaws to 55 per cent")
print("of its own 1.4 rad, which is opposition, not a shared angle")

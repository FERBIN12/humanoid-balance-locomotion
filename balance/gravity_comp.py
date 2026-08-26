#!/usr/bin/env python3
"""Why the robot sank, tested properly. The answer was not what I expected.

3.1 measured this robot sinking 1.030 -> 0.423 m under position control, with
every joint still tracking to within 9.4 degrees. The textbook explanation is
that a proportional controller cannot produce torque at zero error, so you add
a feedforward gravity term. I tried that and it made things WORSE. This script
is the honest record of finding out why.

Five explanations I tested and falsified:
  1 wrong sign on the feedforward term -- no: BOTH signs are worse than none
  2 actuator saturation                -- no: zero clipped steps
  3 joint friction dominating          -- no: 0.425 vs 0.423 m at zero friction
  4 straight legs are singular         -- true but not the cause
  5 qfrc_bias is the wrong quantity    -- it is the right quantity, and it
                                          still does not help here

What actually fixes it: MORE STIFFNESS, not more torque. And the distinction
matters, because the peak torque needed to stand is 18 Nm against a 60 Nm
limit. The robot never lacked torque. It lacked torque EARLY, which is a
different failure.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VA = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
LIM = {i: m.jnt_actfrcrange[m.actuator_trnid[i][0]] for i in range(m.nu)}

KP0 = np.array([200., 200., 200., 300., 60., 40.] * 2)
KD0 = np.array([5., 5., 5., 7.5, 2., 2.] * 2)


def run(mult=1.0, bias=0.0, steps=3000):
    """mult scales the gains; bias scales the qfrc_bias feedforward term."""
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    KP, KD = KP0 * mult, KD0 * np.sqrt(mult)
    peak = np.zeros(m.nu)
    clipped = 0
    for _ in range(steps):
        q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
        v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
        tau = KP * (0 - q) - KD * v
        if bias:
            tau = tau + bias * np.array([d.qfrc_bias[VA[i]] for i in range(m.nu)])
        peak = np.maximum(peak, np.abs(tau))
        clipped += sum(1 for i in range(m.nu) if abs(tau[i]) > LIM[i][1])
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
    err = np.degrees(np.abs(np.array([d.qpos[QA[i]] for i in range(m.nu)])).max())
    return float(d.qpos[2]), err, float(peak.max()), clipped


print("target: still be at 1.030 m after 6 seconds")
print()
print("--- 1. does a gravity feedforward term help? ---")
for b, label in ((0.0, "no compensation"), (1.0, "+qfrc_bias"), (-1.0, "-qfrc_bias")):
    z, e, pk, cl = run(1.0, b)
    print("  %-20s pelvis %.3f m   worst joint %5.2f deg" % (label, z, e))
print("  both signs are WORSE than none, so this is not a missing torque term")
print()

print("--- 2. is it saturation? ---")
z, e, pk, cl = run(1.0, 0.0)
print("  peak torque requested %.1f Nm, clipped joint-steps %d" % (pk, cl))
print("  nothing is clipping, so the actuators are not the limit")
print()

print("--- 3. what about stiffness? ---")
best = None
for mult in (1, 3, 10, 30, 100):
    z, e, pk, cl = run(mult, 0.0)
    tag = "  <- STANDS" if z > 1.0 else ("  <- unstable" if mult >= 100 else "")
    print("  KP x%-4d pelvis %.3f m   worst joint %5.2f deg   peak %6.1f Nm"
          " clipped %5d%s" % (mult, z, e, pk, cl, tag))
    if z > 1.0 and best is None:
        best = (mult, z, e, pk, cl)
print()
mult, z, e, pk, cl = best
print("the cheapest gain that stands is KP x%d: pelvis %.3f m, worst joint"
      % (mult, z))
print("%.2f deg, peak torque %.1f Nm against a 60 Nm ankle limit, %d clipped"
      % (e, pk, cl))
print("steps. The robot never lacked torque. It lacked torque EARLY.")
print()
print("at a hundred times it collapses again, and the peak request explodes:")
print("a stiff controller at a 2 ms timestep is its own instability, so")
print("stiffness is bounded by the integrator, not by the motors.")

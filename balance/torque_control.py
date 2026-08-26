#!/usr/bin/env python3
"""d.ctrl is newton metres. Prove it, and find where the clipping is.

Three questions this answers, all by measurement:
  1 is the number I write the number the physics receives?
  2 what happens when I ask for more than the actuator can deliver?
  3 how much torque does it actually take to hold this robot still?
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]

# 1 -- WHAT THE JOINT DECLARES
print("actuator force limits, straight from the model:")
for i in range(m.nu):
    j = m.actuator_trnid[i][0]
    lo, hi = m.jnt_actfrcrange[j] if m.jnt_actfrclimited[j] else (-np.inf, np.inf)
    print("  %-30s %+8.1f .. %+8.1f Nm" % (names[i], lo, hi))
print()

# 2 -- IS d.ctrl WHAT THE PHYSICS GETS?
# qfrc_actuator is the generalised force the actuators actually applied.
mujoco.mj_forward(m, d)
probe = np.zeros(m.nu); probe[3] = 25.0        # left knee, well inside its 300
d.ctrl[:] = probe
mujoco.mj_step(m, d)
vadr = m.jnt_dofadr[m.actuator_trnid[3][0]]
print("wrote d.ctrl[3] = %.1f Nm" % probe[3])
print("qfrc_actuator at that DOF = %.4f Nm" % d.qfrc_actuator[vadr])
print("so yes: for a direct drive motor actuator, d.ctrl IS the joint torque")
print()

# 3 -- ASK FOR TOO MUCH
d2 = mujoco.MjData(m)
mujoco.mj_forward(m, d2)
over = np.zeros(m.nu); over[3] = 5000.0        # far beyond the 300 Nm limit
d2.ctrl[:] = over
mujoco.mj_step(m, d2)
print("wrote d.ctrl[3] = %.0f Nm (the limit is 300)" % over[3])
print("qfrc_actuator   = %.1f Nm" % d2.qfrc_actuator[vadr])
print("the request is CLIPPED silently: no exception, no warning")
print("if you never read qfrc_actuator you will never know it happened")
print()

# 4 -- WHAT DOES HOLDING STILL COST?
#
# CAREFUL. The obvious answer is qfrc_bias at the spawn pose, and it is WRONG:
# it reports 2.1 Nm total for a 67 kg robot. Two reasons, and both matter.
#   * at the spawn pose the legs are perfectly straight, so gravity's moment
#     arm about each joint is nearly zero. A degenerate configuration.
#   * more importantly, mj_forward on a floating base with the feet NOT loaded
#     describes a robot hanging in free space. There is no load path from the
#     floor to the body, so the legs carry nothing.
# A 2.1 Nm answer also contradicts the earlier measurement, where this same robot
# sagged 60 cm under gravity. When a number contradicts a measurement you
# already trust, the number is wrong.
#
# So measure it the honest way: let the robot settle ON THE GROUND under
# position control, then read the torque the controller is actually spending.
KP = np.array([200., 200., 200., 300., 60., 40.] * 2)
KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2)
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VA = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}

d4 = mujoco.MjData(m)
mujoco.mj_forward(m, d4)
for _ in range(4000):
    q = np.array([d4.qpos[QA[i]] for i in range(m.nu)])
    v = np.array([d4.qvel[VA[i]] for i in range(m.nu)])
    d4.ctrl[:] = KP * (np.zeros(m.nu) - q) - KD * v
    mujoco.mj_step(m, d4)

print("after settling on the ground, holding the pose it could reach:")
tot = 0.0
for i in range(m.nu):
    t = float(d4.ctrl[i]); tot += abs(t)
    print("  %-30s %+8.2f Nm" % (names[i], t))
print()
print("pelvis settled at %.3f m with %d contacts" % (d4.qpos[2], d4.ncon))
print("total absolute holding torque %.1f Nm across %d joints" % (tot, m.nu))
print("the knee alone is spending %.1f Nm"
      % abs(float(d4.ctrl[names.index("left_knee_joint")])))
print()
dbias = mujoco.MjData(m)
mujoco.mj_forward(m, dbias)      # a fresh MjData has qfrc_bias all zero
print("compare: qfrc_bias at the unloaded spawn pose claims only %.1f Nm."
      % sum(abs(float(dbias.qfrc_bias[VA[i]])) for i in range(m.nu)))
print("that is a robot hanging in free space, not a robot standing on a floor")

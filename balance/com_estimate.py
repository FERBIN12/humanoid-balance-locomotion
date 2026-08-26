#!/usr/bin/env python3
"""Where is the centre of mass, and how well can the robot know it?

Three ways to answer, in increasing honesty:
  1 from the model and the true state       -- exact, and unavailable on hardware
  2 from the model and the ENCODERS only    -- what a real robot can compute
  3 from the contact forces (the CoP)       -- a different quantity, often confused

The gap between 1 and 2 is the whole reason state estimation is a field.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VA = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
MASS = float(m.body_mass.sum())


def com_true(data):
    """Mass weighted average of every body's inertial origin."""
    return (m.body_mass[:, None] * data.xipos).sum(0) / MASS


def com_from_encoders(data):
    """What the robot can actually compute: joint angles are known, the FLOATING
    BASE pose is not. So we do forward kinematics with the base at the origin
    and get the CoM in the PELVIS frame, not the world frame."""
    d2 = mujoco.MjData(m)
    d2.qpos[7:] = data.qpos[7:]          # encoders: joint angles only
    d2.qpos[:3] = 0.0
    d2.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # base assumed at identity
    mujoco.mj_forward(m, d2)
    return (m.body_mass[:, None] * d2.xipos).sum(0) / MASS


print("total mass %.2f kg across %d bodies" % (MASS, m.nbody))
print()

KP = np.array([200., 200., 200., 300., 60., 40.] * 2) * 10   # the gain that stands
KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2) * np.sqrt(10.0)

mujoco.mj_forward(m, d)
print("%6s %26s %26s %9s" % ("t", "CoM in world (true)", "CoM in pelvis frame", "|error|"))
for step in range(3000):
    q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
    v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
    d.ctrl[:] = KP * (0 - q) - KD * v
    mujoco.mj_step(m, d)
    if step % 600 == 0 or step == 2999:
        ct = com_true(d)
        ce = com_from_encoders(d)
        # the encoder estimate is in the PELVIS frame; to compare, add the
        # measured pelvis position. On hardware you do not have that either.
        cw = ce + d.qpos[:3]
        print("%6.2f  [%6.3f %6.3f %6.3f]  [%6.3f %6.3f %6.3f]  %9.4f"
              % (d.time, ct[0], ct[1], ct[2], ce[0], ce[1], ce[2],
                 np.linalg.norm(ct - cw)))

print()
ct = com_true(d)
print("the CoM sits %.3f m above the floor with the pelvis at %.3f m"
      % (ct[2], d.qpos[2]))
print("so the CoM is %.3f m BELOW the pelvis: the legs are heavy"
      % (d.qpos[2] - ct[2]))
print()

# --- the part I got wrong, and the measurement that corrected it -------------
# I assumed the CoM RELATIVE TO THE FOOT would be computable from encoders
# alone: the unknown pelvis position appears on both sides and cancels. It
# does. But the base ORIENTATION does not cancel, and a tilt of a degree and a
# half moves the CoM relative to the foot with every joint angle unchanged.
FID = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
truth = com_true(d) - d.xpos[FID]
quat = d.qpos[3:7].copy()
tilt = 2 * np.degrees(np.arccos(min(1.0, abs(float(quat[0])))))
print("CoM relative to the left foot, three ways (base tilt is %.3f deg):" % tilt)
for label, useq in (("encoders only", False), ("encoders + IMU tilt", True)):
    d2 = mujoco.MjData(m)
    d2.qpos[7:] = d.qpos[7:]
    d2.qpos[:3] = 0.0
    d2.qpos[3:7] = quat if useq else [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(m, d2)
    est = (m.body_mass[:, None] * d2.xipos).sum(0) / MASS - d2.xpos[FID]
    print("  %-22s error %8.6f m" % (label, np.linalg.norm(truth - est)))
print()
print("position cancels; ORIENTATION does not. That is what the pelvis IMU is")
print("for: not to tell you where you are, which it cannot, but to supply the")
print("one degree of tilt that turns a 21 mm error into a 16 micron one.")

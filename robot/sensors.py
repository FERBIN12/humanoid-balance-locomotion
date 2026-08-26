#!/usr/bin/env python3
"""What the robot can actually MEASURE, as opposed to what the simulator knows.

The distinction is the whole point. A controller that reads subtree_com or the
contact list is cheating: no real robot has those. It has encoders, an IMU, and
joint torque sensing.
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
for _ in range(600):
    mujoco.mj_step(m, d)

print("sensors declared: %d" % m.nsensor)
for i in range(m.nsensor):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
    a, dim = m.sensor_adr[i], m.sensor_dim[i]
    print("  %-26s %s" % (n, np.round(d.sensordata[a:a + dim], 4)))

print()
print("MEASURABLE on real hardware:")
print("  joint angles      qpos[7:]        %d values" % (m.nq - 7))
print("  joint velocities  qvel[6:]        %d values" % (m.nv - 6))
print("  pelvis ang. rate  IMU             3 values")
print("  gravity direction from the IMU quaternion")
print("  ankle torques     joint sensing   4 values")
print()
print("NOT measurable, simulator only:")
print("  base position     qpos[:3]  -> %s" % np.round(d.qpos[:3], 3))
print("  centre of mass    subtree_com -> %s" % np.round(d.subtree_com[0], 3))
print("  contact list      %d contacts" % d.ncon)
print()
print("this is why the policy in the learned policy uses the GRAVITY DIRECTION rather")
print("than an absolute orientation: gravity is measurable, orientation is not")

#!/usr/bin/env python3
"""Turn an IMU quaternion into the gravity direction, which is what a policy uses.

An IMU gives you orientation relative to gravity, not position in the world. The
gravity direction in the body frame is the third row of the rotation matrix, and
it is the only attitude information a real robot has.
"""
import os, numpy as np, mujoco


def gravity_orientation(q):
    """The official deploy loop's function, term for term."""
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy),
                     -2 * (qz * qy + qw * qx),
                     1 - 2 * (qw * qw + qz * qz)])


m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "pelvis_quat")
adr = m.sensor_adr[sid]

print("%6s  %-26s %-26s" % ("t", "quaternion", "gravity in body frame"))
for step in range(700):
    d.ctrl[:] = 0
    mujoco.mj_step(m, d)
    if step % 140 == 0:
        q = d.sensordata[adr:adr + 4]
        g = gravity_orientation(q)
        print("%6.2f  %-26s %-26s" % (d.time, np.round(q, 3), np.round(g, 3)))
print()
print("upright is gravity = [0, 0, -1] in the body frame")
print("as the robot tips, the first two components grow: that IS the lean signal")

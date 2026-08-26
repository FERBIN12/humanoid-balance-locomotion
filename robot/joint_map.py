#!/usr/bin/env python3
"""Ask for a joint BY NAME. Never hardcode an index."""
import os, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
print("actuators, in model order:")
for i in range(m.nu):
    print("  %2d %s" % (i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)))
print()
i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "left_knee_joint")
print("left_knee_joint -> id %d, qposadr %d, dofadr %d"
      % (i, m.jnt_qposadr[i], m.jnt_dofadr[i]))
print("permute this order and a trained policy still runs, and still falls over")

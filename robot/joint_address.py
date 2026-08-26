#!/usr/bin/env python3
"""Where does a joint's position actually live?

nu actuators, nq position values, and NOT in the same order. Indexing qpos as
7 + actuator_index reads a different joint, which cost me a long detour that
looked exactly like a mechanical jam. See grasp.py note 2.
"""
import os, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
print("nu = %d   nq = %d   nv = %d" % (m.nu, m.nq, m.nv))
print("a free-flyer root eats 7 of nq and 6 of nv, so the offset is NOT uniform")
print()
print("%-34s %5s %8s %8s" % ("actuator", "act i", "qposadr", "7 + i"))
wrong = 0
for i in range(m.nu):
    n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if not n or not any(k in n for k in ("thumb", "index", "pinky")):
        continue
    q = m.jnt_qposadr[m.actuator_trnid[i][0]]
    bad = q != 7 + i
    wrong += bad
    print("%-34s %5d %8d %8d%s" % (n, i, q, 7 + i, "   <- WRONG" if bad else ""))
print()
print("%d of those rows would have read the wrong joint" % wrong)
print("always ask: m.jnt_qposadr[m.actuator_trnid[i][0]]")

#!/usr/bin/env python3
"""The CoM is not the pelvis, and two libraries can disagree about the pose."""
import os, mujoco

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
print("subtree_com[0]  = %.4f %.4f %.4f" % tuple(d.subtree_com[0]))
print("pelvis qpos[:3] = %.4f %.4f %.4f" % tuple(d.qpos[:3]))
print("they differ by %.3f m in z" % abs(d.subtree_com[0][2] - d.qpos[2]))
print()
print("pinocchio at the neutral pose: 0.931 m")
print("MuJoCo at its spawn keyframe:  %.3f m" % d.subtree_com[0][2])
print("a POSE difference, not a library disagreement")
print()
print("heaviest five bodies:")
for i in sorted(range(m.nbody), key=lambda k: -m.body_mass[k])[:5]:
    print("  %-24s %6.2f kg"
          % (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i), m.body_mass[i]))

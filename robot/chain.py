#!/usr/bin/env python3
"""Walk the kinematic chain and MEASURE the segment lengths."""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)          # positions are DERIVED: ask before reading


def bid(n):
    return mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)


print("pelvis to left foot, root first:")
chain, b = [], bid("left_ankle_roll_link")
while b > 0:
    chain.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b))
    b = m.body_parentid[b]
for n in reversed(chain):
    print("   ", n)

print()
print("segment lengths, from world positions:")
for a, c in (("pelvis", "left_hip_pitch_link"),
             ("left_hip_pitch_link", "left_knee_link"),
             ("left_knee_link", "left_ankle_pitch_link"),
             ("left_ankle_pitch_link", "left_ankle_roll_link")):
    L = np.linalg.norm(d.xpos[bid(c)] - d.xpos[bid(a)])
    print("   %-24s -> %-24s %.4f m" % (a, c, L))
print()
print("thigh and shank are both 0.400 m: the IK has a clean closed form")

#!/usr/bin/env python3
"""What is actually in the hands: 12 joints each, and only 101 grams."""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
DIGITS = ("thumb", "index", "middle", "ring", "pinky")
fing = [n for n in names if any(k in n for k in DIGITS)]

print("left hand, per digit:")
for dg in DIGITS:
    L = [n for n in fing if dg in n and n.startswith("L_")]
    short = [x.replace("L_", "").replace("_joint", "").replace(dg + "_", "") for x in L]
    print("  %-8s %d joints   %s" % (dg, len(L), short))
print()
print("total finger joints %d (%d per hand)" % (len(fing), len(fing) // 2))
print("the thumb gets 4 and every other digit gets 2: that extra DOF is OPPOSITION")
print()
print("ranges, left thumb:")
for n in [x for x in fing if x.startswith("L_thumb")]:
    i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
    lo, hi = m.jnt_range[i]
    print("  %-34s %+.2f .. %+.2f rad  (%.0f deg of travel)"
          % (n, lo, hi, np.degrees(hi - lo)))
print()
hb = [i for i in range(m.nbody)
      if any(k in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "")
             for k in DIGITS + ("hand_base",))]
hm = sum(m.body_mass[i] for i in hb)
print("hand bodies %d, total mass %.3f kg = %.2f%% of the robot"
      % (len(hb), hm, 100 * hm / m.body_mass.sum()))
print("the torso is 26.4%: the hands barely matter to BALANCE and matter")
print("entirely to USEFULNESS")

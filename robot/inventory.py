#!/usr/bin/env python3
"""Count the robot, do not assume it.

Two scenes describe DIFFERENT robots:
  scene.xml       12 actuated joints, legs only, what the policy was trained on
  scene_full.xml  51 actuated joints, the whole machine
"""
import os, sys, mujoco

ROOT = os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2")
which = sys.argv[1] if len(sys.argv) > 1 else "scene_full.xml"
m = mujoco.MjModel.from_xml_path(os.path.join(ROOT, which))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, m.njnt)]
groups = {"legs": ("hip", "knee", "ankle"), "arms": ("shoulder", "elbow", "wrist"),
          "fingers": ("thumb", "index", "middle", "ring", "pinky")}
print("%s" % which)
print("  actuated joints %d   bodies %d   mass %.2f kg"
      % (m.nu, m.nbody, m.body_mass.sum()))
counted = 0
for g, keys in groups.items():
    n = [x for x in names if any(k in x for k in keys)]
    counted += len(n)
    print("  %-8s %2d" % (g, len(n)))
print("  %-8s %2d" % ("other", len(names) - counted))

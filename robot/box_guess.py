#!/usr/bin/env python3
"""What a uniform box approximation costs you, on the real torso."""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
i = int(np.argmax(m.body_mass))
name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)
mass, I = m.body_mass[i], m.body_inertia[i]
d, w, h = 0.20, 0.30, 0.25          # a plausible eyeball of the torso
box = mass / 12 * np.array([w * w + h * h, d * d + h * h, d * d + w * w])
print("body: %s, %.2f kg" % (name, mass))
print("  real inertia %s" % np.round(I, 4))
print("  box guess    %s   (%.2f x %.2f x %.2f m)" % (np.round(box, 4), d, w, h))
print("  ratio        %s" % np.round(I / box, 2))
print()
print("  worst axis is out by %.2fx" % max(I / box))
print("  and the AXIS ORDER differs: box smallest is axis %d, real smallest is axis %d"
      % (int(np.argmin(box)), int(np.argmin(I))))

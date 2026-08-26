#!/usr/bin/env python3
"""Load the H1-2 in MuJoCo and print what the model actually contains."""
import os
import mujoco

SCENE = os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)

print("nq = %d   (12 joints + 7 floating base)" % m.nq)
print("nv = %d   (12 joints + 6, a quaternion has 4 numbers but 3 DOF)" % m.nv)
print("nu = %d   actuators" % m.nu)
print("nbody = %d" % m.nbody)
print("timestep = %.4f s  ->  %.0f Hz" % (m.opt.timestep, 1 / m.opt.timestep))
print("total mass = %.2f kg" % m.body_mass.sum())
print("CoM at spawn = %.3f m" % d.subtree_com[0][2])

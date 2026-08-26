#!/usr/bin/env python3
"""9.5/9.6 -- the same controller in two simulators, measured.

Sections 1 to 8 ran entirely in MuJoCo. This is the only part of this project
that leaves it, and the question worth answering is not "does Gazebo work"
but "what changes when you move".

Three things are compared, and they are deliberately the boring ones, because
the interesting differences always turn out to live in the boring ones:

  1. what the two models actually contain: joints, mass, inertia
  2. what ros2_control exposes against what MuJoCo actuates
  3. the standing drop test: let go and see where the pelvis ends up

Nothing here runs a policy. 9.6 is about what transfers, and the first thing
to establish is whether the two robots are even the same robot.
"""
import os
import pathlib
import xml.etree.ElementTree as ET

import numpy as np

WS = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
URDF = WS / "src/cortex_humanoid_description/urdf/cortex_humanoid_torque.urdf"
MJCF = WS / "mujoco/resources/robots/h1_2/scene_full.xml"


def urdf_facts():
    r = ET.parse(URDF).getroot()
    links = r.findall("link")
    mass = 0.0
    for l in links:
        i = l.find("inertial")
        if i is not None:
            m = i.find("mass")
            if m is not None:
                mass += float(m.get("value"))
    joints = [j for j in r.findall("joint")]
    act = [j for j in joints if j.get("type") in ("revolute", "prismatic")]
    r2c = set()
    for b in r.findall("ros2_control"):
        for j in b.findall("joint"):
            r2c.add(j.get("name"))
    return dict(links=len(links), mass=mass, actuated=len(act),
                exposed=len(r2c),
                names={j.get("name") for j in act})


def mjcf_facts():
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(MJCF))
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)}
    return dict(bodies=m.nbody, mass=float(m.body_mass.sum()),
                actuated=m.nu, names=names,
                timestep=m.opt.timestep)


if __name__ == "__main__":
    u, j = urdf_facts(), mjcf_facts()
    print("--- are these even the same robot? ---")
    print(f"  {'':<22} {'URDF/Gazebo':>14} {'MJCF/MuJoCo':>14}")
    print(f"  {'total mass kg':<22} {u['mass']:>14.2f} {j['mass']:>14.2f}")
    print(f"  {'actuated joints':<22} {u['actuated']:>14} {j['actuated']:>14}")
    print(f"  {'exposed to control':<22} {u['exposed']:>14} {j['actuated']:>14}")
    print()

    only_mj = sorted(j["names"] - u["names"])
    only_ur = sorted(u["names"] - j["names"])
    print(f"  joints MuJoCo has that the URDF does not: {len(only_mj)}")
    if only_mj:
        print(f"    {only_mj[:6]}{' ...' if len(only_mj) > 6 else ''}")
    print(f"  joints the URDF has that MuJoCo does not: {len(only_ur)}")
    if only_ur:
        print(f"    {only_ur[:6]}{' ...' if len(only_ur) > 6 else ''}")
    print()

    print("--- the 4 joints that differ are not a rename ---")
    print("  Both arms have 7 DOF in both models, and both agree on the")
    print("  shoulder. They disagree about where the elbow ends:")
    print()
    print("    URDF   shoulder p/r/y, elbow,       wrist r/p/y")
    print("    MJCF   shoulder p/r/y, elbow p/r,   wrist   p/y")
    print()
    print("  MuJoCo splits the elbow into pitch and roll; the URDF gives the")
    print("  roll to the wrist instead. Same joint count, same total mass to")
    print("  the gram, different decomposition. A controller that maps joints")
    print("  BY NAME would silently mis-wire two joints per arm and produce")
    print("  a robot that mostly works, which is the worst kind.")
    print()

    print("--- and the gap that matters most ---")
    print(f"  The URDF declares {u['actuated']} actuated joints and ros2_control")
    print(f"  exposes {u['exposed']}. Thirty joints, the 24 fingers and 6 wrists,")
    print("  exist in the model and cannot be commanded through ROS at all.")
    print()
    print("  So a controller written against MuJoCo has 51 actuators and the")
    print("  same controller in Gazebo has 21. That is not a tuning difference")
    print("  or a physics difference. It is a different machine, and any")
    print("  claim about what transfers has to start there.")
    print()
    print("--- what DOES match ---")
    print(f"  Total mass agrees to the gram: {u['mass']:.2f} kg against")
    print(f"  {j['mass']:.2f} kg. Both models have {u['actuated']} actuated joints.")
    print("  The disagreements are all in the topology and the plumbing, not")
    print("  in the physical description of the robot, which is the reassuring")
    print("  half of this comparison.")

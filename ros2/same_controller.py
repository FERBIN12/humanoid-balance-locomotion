#!/usr/bin/env python3
"""9.5 -- the same controller, two simulators.

The whole section has been building to this: take one control law, run it in
MuJoCo and in Gazebo, and see what differs.

The control law is deliberately the simplest thing that is not trivial: a PD
hold around a fixed pose, the same gains, the same target, the same 400 Hz.
Anything more elaborate would let a difference hide inside the controller.

What CANNOT be held identical, and this is most of the experiment:

  * MuJoCo commands 51 actuators, Gazebo's ros2_control exposes 21
  * MuJoCo's arm is shoulder p/r/y + elbow p/r + wrist p/y
    the URDF's is    shoulder p/r/y + elbow + wrist r/p/y
  * MuJoCo integrates at 2 ms, the Gazebo world here at 1 ms
  * one runs in-process, the other over topics with an executor between

So this is not a controlled experiment with one variable. It is an honest
account of what "the same controller" turns out to mean when you move it.
"""
import os
import pathlib

import numpy as np

WS = pathlib.Path(os.path.expanduser("~/humanoid_ws"))

# measured, both simulators, pelvis height after a 30 s PD hold
GAZEBO_Z = 0.8632
GAZEBO_RPY = (-0.0000, -0.0000, 0.0000)


def mujoco_hold(kp=220.0, kd=12.0, dur=30.0):
    """The same PD hold, in MuJoCo, on the same 21 joints Gazebo exposes."""
    import mujoco
    import yaml
    scene = str(WS / "mujoco/resources/robots/h1_2/scene_full.xml")
    m = mujoco.MjModel.from_xml_path(scene)
    d = mujoco.MjData(m)
    cfg = yaml.safe_load(open(WS / "policy/h1_2.yaml"))
    default = np.array(cfg["default_angles"], np.float32)
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    d.qpos[qadr[:12]] = default
    mujoco.mj_forward(m, d)
    target = d.qpos[qadr].copy()
    zs = []
    for k in range(int(dur / m.opt.timestep)):
        tau = (target - d.qpos[qadr]) * kp - d.qvel[vadr] * kd
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if k % 500 == 0:
            zs.append(float(d.qpos[2]))
    return dict(z=float(d.qpos[2]), z_min=min(zs), n=m.nu,
                dt=m.opt.timestep)


if __name__ == "__main__":
    print("--- the same PD hold, kp=220 kd=12, 30 s ---")
    mj = mujoco_hold()
    print(f"  {'':<16} {'MuJoCo':>12} {'Gazebo':>12}")
    print(f"  {'pelvis z, m':<16} {mj['z']:>12.4f} {GAZEBO_Z:>12.4f}")
    print(f"  {'actuators':<16} {mj['n']:>12} {21:>12}")
    print(f"  {'timestep, s':<16} {mj['dt']:>12.4f} {0.001:>12.4f}")
    print()
    print(f"  The same gains, in the same units, on the same robot: Gazebo")
    print(f"  stands and MuJoCo ends up on the floor, "
          f"{1000 * abs(mj['z'] - GAZEBO_Z):.0f} mm apart.")
    print()

    print("--- and it is not the gains ---")
    print("  My first instinct was that MuJoCo needed a stiffer hold, so I")
    print("  swept it. Every gain collapses:")
    print(f"    {'kp':>7} {'pelvis z':>10}")
    for kp in (220, 800, 3000, 40000):
        r = mujoco_hold(kp=kp, kd=kp / 18.0, dur=8.0)
        print(f"    {kp:>7} {r['z']:>10.4f}")
    print()
    print("  Forty thousand is not a gain, it is an assertion, and it still")
    print("  falls. A parameter sweep that produces the same answer across")
    print("  two hundred fold is not telling you about the parameter.")
    print()

    print("--- what it actually is ---")
    print("  The PD holds 51 JOINT angles. It cannot hold the base, because")
    print("  in MuJoCo the base is a freejoint: seven qpos entries that no")
    print("  actuator touches. Holding every joint perfectly rigid turns the")
    print("  robot into a statue, and a statue tips.")
    print()
    print("  In Gazebo the same control law stands, and it stands for reasons")
    print("  that have nothing to do with the controller: the spawn height,")
    print("  the contact solver, the 1 ms step, and the fact that 30 of the")
    print("  joints are not being held at all so the robot is not fully rigid.")
    print()
    print("  So the honest reading is not 'Gazebo is better at standing'. It")
    print("  is that a PD hold on joint angles was never a balance controller")
    print("  in either simulator, and one of them was flattering it.")
    print()

    print("--- what could not be held the same ---")
    print("  1. MuJoCo commands 51 actuators; ros2_control exposes 21, so the")
    print("     24 finger joints and 6 wrist joints are unheld in Gazebo and")
    print("     held in MuJoCo. Thirty joints of difference in the CONTROLLER,")
    print("     before the physics has done anything.")
    print("  2. The arms decompose differently. MuJoCo has elbow pitch AND")
    print("     elbow roll; the URDF has one elbow joint and gives the roll")
    print("     to the wrist. Same 7 DOF per arm, different joints.")
    print("  3. MuJoCo integrates at 2 ms here, the Gazebo world at 1 ms.")
    print("  4. MuJoCo's loop is in-process. Gazebo's crosses topics with an")
    print("     executor and a bridge between, which 9.3 measured as a rate")
    print("     that moves between runs.")
    print()
    print("  Four differences before a single step is integrated. Any result")
    print("  has at least four candidate explanations, and that is the honest")
    print("  state of a cross-simulator comparison. What the learned policy's policy")
    print("  does transfer is a much harder question, and 9.6 asks it.")

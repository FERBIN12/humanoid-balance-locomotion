#!/usr/bin/env python3
"""9.7 -- toward hardware: what the deploy path would actually require.

This experiment does not run on hardware. There is no H1-2 on my desk, and a
experiment that pretended otherwise would be worse than useless.

What it can honestly do is enumerate, from the code that exists, exactly what
would have to change and what would have to be verified. Every item below is
derived from something measured in sections 1 through 9, not from a general
list of good practices.
"""
import os
import pathlib

ITEMS = [
    dict(what="the effort state interface",
         now="gz_ros2_control echoes back the commanded effort",
         hw="a current sensor or strain gauge, which will DISAGREE",
         why="9.4: a controller closing on simulated effort closes on its "
             "own output and is unconditionally stable in sim",
         risk="high"),
    dict(what="the 30 unexposed joints",
         now="ros2_control exposes 21 of 51; fingers and wrists are limp",
         hw="real hardware exposes what it exposes, and it may differ again",
         why="9.2, 9.6: 'the same controller' already means two things",
         risk="high"),
    dict(what="the loop rate",
         now="369 to 2273 Hz depending on the real time factor",
         hw="a fixed clock, and a missed deadline is a fault not a slowdown",
         why="9.3: sim time hid the fact that the rate was never pinned",
         risk="high"),
    dict(what="the arm joint decomposition",
         now="MuJoCo elbow p/r vs URDF elbow + wrist roll",
         hw="whatever the manufacturer's driver publishes",
         why="9.6: same DOF count, four of seven joints do not line up",
         risk="high"),
    dict(what="the armature",
         now="the shipped MJCF had ZERO on every DOF; we set 0.01",
         hw="real rotor and gearbox inertia, per joint, from a datasheet",
         why="the robot model: gram-scale finger links diverged at 6842 rad/s "
             "without it",
         risk="medium"),
    dict(what="contact and friction",
         now="mu = 1.0 everywhere, chosen not measured",
         hw="a floor, shoes, and dust",
         why="8.2: slipping needs 45 degrees at mu=1, which is why it never "
             "bound in simulation",
         risk="medium"),
    dict(what="the fall behaviour",
         now="the robot falls freely and we reset the sim",
         hw="a 67 kg machine hitting a floor, once",
         why="8.7: six failure modes, four of which end on the floor",
         risk="the whole project"),
]


if __name__ == "__main__":
    print("--- what the deploy path would require ---")
    print("  Nothing here runs on hardware. This is what the code says would")
    print("  have to change, item by item, derived from measurements in this")
    print("  project rather than from a generic checklist.")
    print()
    for it in ITEMS:
        print(f"  {it['what'].upper()}   [{it['risk']}]")
        print(f"    in sim:   {it['now']}")
        print(f"    on hw:    {it['hw']}")
        print(f"    because:  {it['why']}")
        print()
    hi = [i for i in ITEMS if i["risk"] == "high"]
    print(f"--- {len(hi)} of {len(ITEMS)} are high risk, and they share a shape ---")
    print("  Every one of them is a case where the simulator supplied")
    print("  something the hardware will not: a state that was really a")
    print("  command, a clock that stretched, a joint set that happened to")
    print("  match, a rate that was never pinned. None of them is a physics")
    print("  error. They are all interface fictions.")
    print()
    print("  Which suggests the useful discipline: for every number your")
    print("  controller reads, ask what physical device would produce it on")
    print("  hardware. If the answer is 'nothing, the simulator computes it")
    print("  from my own output', you have found one.")

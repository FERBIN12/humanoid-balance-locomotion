#!/usr/bin/env python3
"""10.2 -- the world the mission runs in.

The capstone project has to contain terrain the robot can actually handle and
terrain it cannot, or the run proves nothing. The terrain work measured exactly where
those boundaries are, so the world is built from those numbers rather than
from a designer's taste:

    a 4 deg ramp      8.2: climbs 2.68 m at 0.5 m/s, all of it at 0.9
    a 20 mm step      8.4: marginal, 2 of 3 seeds clean
    a 300 mm gap      8.4: crosses reliably, falls into 375
    a 6 deg ramp      8.1: stalls completely at 0.5 m/s

The last one is deliberate. A capstone project with nothing impossible in it is
a demo reel, and the honest thing is to include an obstacle the robot cannot
pass and say so before the run rather than after.
"""
import os
import sys
import pathlib

sys.path.insert(0, os.path.expanduser("~/humanoid_ws/project/code/section-08"))
import terrain as T                                        # noqa: E402

# each stage: (name, builder, argument, what the terrain work measured)
STAGES = [
    ("flat approach", "flat", 0.0, "steady state before anything happens"),
    ("4 deg ramp", "ramp", 4.0, "8.2: 2.68 m at 0.5, all 6 m at 0.9"),
    ("20 mm step", "steps", 0.020, "8.4: marginal, 2 of 3 seeds clean"),
    ("300 mm gap", "gap", 0.300, "8.4: crosses; 375 mm does not"),
    ("6 deg ramp", "ramp", 6.0, "8.1: stalls at 0.5, climbs at 0.9"),
]


def build(kind, arg):
    return {"flat": lambda a: T.flat(),
            "ramp": lambda a: T.ramp(a),
            "steps": lambda a: T.steps(a),
            "gap": lambda a: T.gap(a)}[kind](arg)


if __name__ == "__main__":
    print("--- the mission project, built from the terrain work's numbers ---")
    print(f"  {'stage':<16} {'geometry':<10} {'what 8.x measured'}")
    for name, kind, arg, why in STAGES:
        m = build(kind, arg)
        print(f"  {name:<16} {kind:<10} {why}")
    print()
    print("  Every stage loads and every one has a measured expectation")
    print("  attached, which is the difference between a project and a set")
    print("  of obstacles.")
    print()
    print("--- and one of them is impossible ---")
    print("  The 6 degree ramp at 0.5 m/s stalls. 8.1 measured 0.09 m of")
    print("  climb in 25 seconds, and the robot stays upright the whole time")
    print("  looking exactly like a healthy walking machine.")
    print()
    print("  It is in this project on purpose. A capstone with nothing")
    print("  impossible in it is a demo reel, and the useful version tells")
    print("  you where the wall is BEFORE the run rather than after.")

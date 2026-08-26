#!/usr/bin/env python3
"""8.7 -- every way this robot falls, in one place, and what they share.

The terrain work has produced a lot of failures. They are not all the same failure,
and the point of collecting them is that the differences are the content.

Six modes, each with the number that defines it:

  STALL      walks to the foot of a slope and stops, upright, forever
  RUN OUT    climbs a slope with each step shorter than the last
  TRIP       catches a riser, ground arrives ahead of the gait phase
  DROP       steps into a hole with nothing under the foot
  SHOVE      a lateral push exceeds what the ankle can absorb
  EDGE       walks off the side of terrain, which is my geometry, not the robot

The last one is in the list deliberately. Three of the confident wrong answers
in this section came from the robot leaving terrain I had built too small, and
a failure gallery that only lists the robot's failures teaches half a lesson.
"""
import os
import pathlib

import numpy as np

MODES = [
    dict(name="STALL", where="6 deg slope at 0.5 m/s",
         number="0.09 m climbed in 25 s",
         mech="propulsion: gravity along the slope is 69 N and never stops",
         detect="height gained, NOT a fall flag or path length",
         experiment="8.1"),
    dict(name="RUN OUT", where="10 deg slope at 0.9 m/s",
         number="2.96 m, each step shorter than the last",
         mech="the same propulsion limit, met gradually rather than at once",
         detect="progress per step, which decays instead of stopping",
         experiment="8.3"),
    dict(name="TRIP", where="25 mm step",
         number="fails at 25 mm with 64 mm of swing height",
         mech="timing: the ground arrives ahead of the gait phase",
         detect="still standing at the end, NOT distance travelled",
         experiment="8.4"),
    dict(name="DROP", where="375 mm gap",
         number="crosses 350 mm, falls into 375",
         mech="stride length, not foot length: it steps OVER holes",
         detect="pelvis below the local ground height",
         experiment="8.4"),
    dict(name="SHOVE", where="lateral push on the flat",
         number="223 N sideways against 409 forward",
         mech="the foot is 110 mm wide and 240 mm long",
         detect="bisected boundary, never a single magnitude",
         experiment="8.6"),
    dict(name="EDGE", where="my own terrain",
         number="veers to |y| = 4.2 m in 25 s",
         mech="none: this is the simulation, not the robot",
         detect="refuse the run if it came within 0.6 m of an edge",
         experiment="8.1, 8.2, 8.6"),
]


if __name__ == "__main__":
    print("--- six ways this robot ends up on the floor ---")
    print("  Or, in two of the six cases, does not end up on the floor at all,")
    print("  which is exactly why they are worth separating.")
    print()
    for m in MODES:
        print(f"  {m['name']:<8} {m['where']}")
        print(f"           {m['number']}")
        print(f"           mechanism: {m['mech']}")
        print(f"           detect by: {m['detect']}   ({m['experiment']})")
        print()

    print("--- what they have in common ---")
    print("  Two of the six are not falls. STALL and RUN OUT leave the robot")
    print("  upright, pelvis at one metre, stepping in place. Every safety")
    print("  check you would normally write reports a healthy machine, and the")
    print("  task has completely failed. That is the single most useful thing")
    print("  in this section: the failure you should most fear is the one your")
    print("  monitoring is blind to.")
    print()
    special = [m for m in MODES if "NOT" in m["detect"]]
    print(f"  {len(special)} of the {len(MODES)} needed a DIFFERENT metric to "
          f"see at all:")
    for m in special:
        print(f"    {m['name']:<8} {m['detect']}")
    print()
    print("  And one of the six is not the robot's fault. EDGE is my terrain")
    print("  being narrower than the robot's own lateral drift, and it")
    print("  produced three confident wrong answers before I put a refusal")
    print("  inside the measurement instead of a comment beside it.")
    print()

    print("--- the split that matters ---")
    print(f"  {'mode':<9} {'robot upright?':>15} {'task done?':>12}")
    for m, up, done in ((MODES[0], "YES", "no"), (MODES[1], "YES", "partly"),
                        (MODES[2], "no", "no"), (MODES[3], "no", "no"),
                        (MODES[4], "no", "no"), (MODES[5], "no", "n/a")):
        print(f"  {m['name']:<9} {up:>15} {done:>12}")
    print()
    print("  A fall detector sees rows three through six and misses the first")
    print("  two entirely. A distance metric sees rows one and two and calls")
    print("  row three a success, because a robot that topples forward past a")
    print("  staircase travels exactly as far as one that climbed it.")
    print()
    print("  There is no single number that catches all six. That is not a")
    print("  gap in this project, it is the actual state of the problem, and")
    print("  the practical answer is that every experiment needs a metric")
    print("  chosen for the failure it is capable of producing.")

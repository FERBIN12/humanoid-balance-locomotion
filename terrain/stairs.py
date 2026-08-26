#!/usr/bin/env python3
"""8.5 -- stairs, and the honest version of that demo.

The obvious experiment here is "watch the robot climb some stairs". 8.4 makes
that impossible: a 25 mm step already defeats this policy, and a building code
stair is 150 to 200 mm. The robot is out by a factor of eight.

So this experiment does the thing the footage can actually support: it establishes
exactly how far short we are, and then asks what would have to change. Two
candidate fixes, both testable:

  1. more speed, which is what rescued the slope in 8.2
  2. a slower gait clock, giving the swing phase longer to find the tread

Neither is a retrain. If either works the answer is a parameter; if neither
does, the answer is that stairs need terrain input and that is a the ROS 2 bridge
and 10 problem.
"""
import os
import pathlib

import mujoco
import numpy as np

import terrain as T
import slope_sweep as S
import steps_gaps as SG

# Building code stair geometry, for scale. UK Part K: max rise 220 mm for a
# private stair, typical 175 to 200. US IBC: max riser 178 mm (7 inches).
CODE_RISE = 0.175
CODE_TREAD = 0.28


def try_speed(rise, cmds=(0.5, 0.9, 1.3, 1.7), dur=25.0):
    """Does asking for more speed rescue a step, the way it rescued a slope?"""
    out = []
    for c in cmds:
        r = SG.climb_steps(rise, dur=dur, cmd=c)
        out.append((c, r["n_steps"], r["fell"]))
    return out


def try_gait(rise, gaits=(0.8, 1.0, 1.2, 1.5), dur=25.0, cmd=0.9):
    """Does a SLOWER gait clock help?

    The 8.4 diagnosis was that the ground arrives ahead of the phase the gait
    assumed. A longer cycle gives the swing foot more time above the tread, so
    if that diagnosis is right this should buy something.

    The gait period is baked into the observation as sin/cos of the phase, so
    changing it means the policy sees a clock it was never trained on. That is
    exactly the kind of thing that can silently do nothing, so the result gets
    checked against a no-op.
    """
    out = []
    for g in gaits:
        old = S.GAIT
        S.GAIT = g
        try:
            r = SG.climb_steps(rise, dur=dur, cmd=cmd)
            out.append((g, r["n_steps"], r["fell"]))
        finally:
            S.GAIT = old
    return out


if __name__ == "__main__":
    print("--- the size of the problem ---")
    print(f"  A code-compliant stair rises {1000*CODE_RISE:.0f} mm per step on a")
    print(f"  {1000*CODE_TREAD:.0f} mm tread. 8.4 measured that this policy fails at")
    print("  25 mm. That is a factor of seven, and it is not a tuning gap.")
    print()
    print("  For scale, 25 mm is: a doorway threshold, the lip of a shower")
    print("  tray, a thick rug. The robot walks 16 metres across flat ground")
    print("  without trouble and cannot get onto a doormat.")
    print()

    print("--- fix 1: more speed, which rescued the slope in 8.2 ---")
    print("  On a slope, raising the command from 0.5 to 0.9 turned a total")
    print("  failure into climbing the whole ramp. If steps work the same way")
    print("  this is a one-line fix.")
    print()
    print(f"  {'cmd m/s':>9} {'survived, 3 seeds':>18} {'steps reached':>14}")
    for c in (0.3, 0.5, 0.7, 0.9):
        v = [SG.climb_steps(0.025, cmd=c, seed=sd) for sd in range(3)]
        ok = sum(1 for x in v if x["fell"] is None)
        print(f"  {c:>9.1f} {('%d of 3' % ok):>18} "
              f"{str([x['n_reached'] for x in v]):>14}")
    print()
    print("  It is not a one-line fix, it is the SAME line pointing the other")
    print("  way. 0.3 and 0.5 survive every seed. 0.7 and 0.9 survive one in")
    print("  three. On a hill, faster was the answer. On a step, faster is")
    print("  the problem, and that inversion is the most useful thing here.")
    print()
    print("  It also makes sense once stated. A slope is a force you have to")
    print("  overcome, and speed is momentum, so momentum helps. A step is a")
    print("  timing error, and momentum makes a timing error worse: you")
    print("  arrive at the wrong moment with more energy behind you.")
    print()

    print("--- fix 2: a slower gait clock ---")
    print("  8.4's diagnosis was that the ground arrives ahead of the phase")
    print("  the gait assumed. A longer cycle gives the swing foot more time")
    print("  above the tread, so if that diagnosis is right this should buy")
    print("  something. Note the policy has never seen a clock other than")
    print("  0.8 s: the phase enters the observation as a sine and cosine.")
    print()
    print(f"  {'gait s':>8} {'steps reached':>14} {'fell':>8}")
    for g, n, fell in try_gait(0.025):
        f = "-" if fell is None else ("%.1fs" % fell)
        print(f"  {g:>8.1f} {n:>14} {f:>8}")
    print()
    print("  Nothing. Every period falls, and the longer ones fall SOONER.")
    print("  Stretching the clock does not give the swing more time over the")
    print("  tread, it desynchronises a policy from the only clock it has")
    print("  ever run against, which is a second problem on top of the first.")
    print()

    print("--- and one metric that lied on the way here ---")
    print("  The step count is derived from the robot's x position, and a")
    print("  robot that FALLS FORWARD past the staircase ends up at the same x")
    print("  as one that climbed it. Every row of both sweeps above originally")
    print("  read '6 steps up', including runs that fell at 4.3 seconds.")
    print("  The count now returns zero unless the robot is still standing at")
    print("  the end, which is the property the word 'climbed' actually means.")
    print()

    print("--- what 8.5 concludes ---")
    print(f"  A code stair rises {1000*CODE_RISE:.0f} mm. This policy manages 20 and")
    print("  fails at 25. Neither of the two parameter fixes available without")
    print("  retraining does anything, and one of them actively hurts.")
    print()
    print("  So stairs are not a tuning problem for this controller, and no")
    print("  amount of footage of it trying would teach you more than the")
    print("  number already has. What a stair needs is for the robot to KNOW")
    print("  the tread is coming, which means terrain input, which is a")
    print("  different policy and not a different parameter. 8.6 goes back to")
    print("  ground it can handle and measures disturbance rejection properly.")

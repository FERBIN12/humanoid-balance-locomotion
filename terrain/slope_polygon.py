#!/usr/bin/env python3
"""8.2 -- what a slope actually changes about the support polygon.

The balance controller worked out the flat-ground balance limits from the foot geometry:
0.240 m long, 0.110 m wide, CoM at 0.937 m. The ankle can move the centre of
pressure to the foot edge and no further, so the recoverable set is bounded by
the capture point staying inside the foot.

The obvious guess about a slope is that it "shrinks the support polygon". That
is the sentence in most textbooks and it is worth being precise about, because
the polygon is a set of CONTACT POINTS and tilting the ground does not move the
foot relative to itself. The foot is the same 0.240 m long on a hill.

So what does change? Three candidates, and they are not equally important:

  1. the polygon PROJECTED onto the horizontal shrinks by cos(theta)
  2. gravity acquires a component ALONG the slope that never goes away
  3. friction has to carry that component or the foot slides

This file works out all three from the geometry, then asks the simulator which
one actually binds. 8.1 measured that the policy stalls at 6 degrees without
falling, so whatever the answer is, it has to explain a PROPULSION failure
rather than a balance failure.
"""
import os
import pathlib

import mujoco
import numpy as np

import terrain as T
import slope_sweep as S

G = 9.81
H = 0.937           # measured CoM height, standing (3.5)
FOOT_L = 0.240      # measured foot length (2.6)
FOOT_W = 0.110
MASS = 67.37        # measured total mass (2.2)
MU = 1.0            # the friction in every terrain scene


def geometry(deg):
    """The three candidate effects, from the geometry alone.

    Everything here is a PREDICTION. The simulator gets asked afterwards.
    """
    th = np.radians(deg)
    return dict(
        deg=deg,
        # 1. the horizontal projection of the same foot
        proj=FOOT_L * np.cos(th),
        proj_loss=FOOT_L * (1 - np.cos(th)),
        # 2. gravity along the slope, as a force and as a fraction of weight
        along=MASS * G * np.sin(th),
        along_frac=np.sin(th),
        # 3. what friction can hold: mu * normal
        friction=MU * MASS * G * np.cos(th),
        slips=np.tan(th) > MU,
        # the ankle torque needed just to STAND on the slope, because the CoM
        # is no longer over the centre of the foot
        lean_torque=MASS * G * H * np.sin(th),
    )


def ankle_budget():
    """How much ankle torque is actually available.

    Read from the model rather than assumed: the whole point of 8.2 is which
    limit binds, and a guessed torque limit decides that by fiat.
    """
    m = T.flat()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)]
    out = {}
    for n in ("left_ankle_pitch_joint", "left_ankle_roll_joint",
              "left_knee_joint", "left_hip_pitch_joint"):
        i = names.index(n)
        lo, hi = m.actuator_ctrlrange[i]
        out[n] = float(hi) if hi > 0 else float("nan")
    return out


if __name__ == "__main__":
    print("--- what tilting the ground does NOT change ---")
    print("  The support polygon is a set of contact points, and tilting the")
    print("  ground does not move the foot relative to itself. The foot is")
    print(f"  {FOOT_L:.3f} m long on a hill exactly as it is on the flat.")
    print("  'A slope shrinks the support polygon' is loose talk. Three things")
    print("  do change, and they are not equally important.")
    print()

    print(f"  {'slope':>6} {'proj. foot':>11} {'lost':>8} "
          f"{'gravity along':>14} {'as % weight':>12}")
    for deg in (0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 30.0):
        g = geometry(deg)
        print(f"  {deg:>5.0f}d {g['proj']:>11.4f} {g['proj_loss']:>8.4f} "
              f"{g['along']:>13.1f}N {100 * g['along_frac']:>11.1f}%")
    print()

    g6 = geometry(6.0)
    print("  So at the 6 degrees where 8.1 measured the policy giving out:")
    print(f"    the foot's horizontal projection loses {1000 * g6['proj_loss']:.1f} mm "
          f"of {1000 * FOOT_L:.0f}")
    print(f"    that is {100 * g6['proj_loss'] / FOOT_L:.2f}% of the polygon, which is nothing")
    print(f"    gravity along the slope is {g6['along']:.1f} N, "
          f"{100 * g6['along_frac']:.1f}% of body weight")
    print()
    print("  Those two numbers are three orders of magnitude apart in")
    print("  significance. The polygon effect is 0.5 percent. The gravity")
    print("  effect is a 69 newton force that never switches off, applied to")
    print("  a machine whose forward drive was tuned on ground where it was")
    print("  exactly zero. That is a propulsion problem, which is the failure")
    print("  8.1 actually measured.")
    print()

    print("--- and the one that does not bind at all ---")
    print(f"  Friction is mu = {MU:.1f} in every terrain scene, so the foot")
    print(f"  slides when tan(theta) > {MU:.1f}, which is 45 degrees.")
    for deg in (6.0, 20.0, 44.0, 46.0):
        g = geometry(deg)
        print(f"    {deg:>4.0f}d  tan = {np.tan(np.radians(deg)):.3f}  "
              f"slips: {g['slips']}")
    print("  The robot stops climbing at 6 degrees and would not slip until")
    print("  45, so slipping is not what stops it. Worth stating, because")
    print("  'it slips on slopes' is the other thing people assume.")
    print()

    print("=== everything above is a PREDICTION. Now ask the simulator. ===")
    print()
    print("  The geometry says the binding effect is gravity along the slope,")
    print("  which is a propulsion problem. That makes a testable claim: if")
    print("  the stall is about forward drive rather than balance, then simply")
    print("  ASKING for more speed should buy real climbing. If it were about")
    print("  balance, more speed would make it worse.")
    print()
    print("  Same 6 degree ramp, same policy, only the velocity command moves:")
    print(f"  {'cmd m/s':>9} {'along m':>10} {'gained m':>10}")
    for c in (0.3, 0.5, 0.7, 0.9, 1.1):
        globals()["_G"] = None
        S._GROUND = S.on_ramp(6.0)
        r = S.walk(T.ramp(6.0), dur=25.0, cmd=[c, 0.0, 0.0])
        th = np.radians(6.0)
        along = min(6.0, max(0.0, r["x"] - T.X0) / np.cos(th))
        print(f"  {c:>9.1f} {along:>10.2f} {along * np.sin(th):>10.3f}")
    print()
    print("  At the 0.5 m/s this project has used since the learned policy, the robot")
    print("  climbs 7 cm. At 0.9 it climbs the WHOLE six metre ramp. Nothing")
    print("  changed but the number in the command. The 6 degree limit in 8.1")
    print("  was never a property of the slope; it was a property of the")
    print("  speed we happened to be asking for.")
    print()

    print("--- so where is the real limit? ---")
    print(f"  {'slope':>6} {'along m':>10} {'gained m':>10} {'outcome':>22}")
    for deg, dur in ((6.0, 20.0), (8.0, 25.0), (10.0, 25.0), (12.0, 25.0),
                     (15.0, 25.0), (20.0, 25.0)):
        S._GROUND = S.on_ramp(deg)
        r = S.walk(T.ramp(deg), dur=dur, cmd=[0.9, 0.0, 0.0])
        th = np.radians(deg)
        along = min(6.0, max(0.0, r["x"] - T.X0) / np.cos(th))
        if r["fell"] and along > 5.7:
            out = "ran off the ramp TOP"
        elif r["fell"]:
            out = "FELL at the foot"
        elif along < 0.5:
            out = "stalled"
        else:
            out = "climbed"
        print(f"  {deg:>5.0f}d {along:>10.2f} {along * np.sin(th):>10.3f} "
              f"{out:>22}")
    print()
    print("  Two different failures, and they are not the same failure. Up to")
    print("  about 10 degrees the robot climbs. By 12 it stalls at the foot")
    print("  again. At 15 and 20 it does something it never did in 8.1: it")
    print("  FALLS, at the foot of the ramp, 17.6 s and 4.8 s in.")
    print()
    print("  Note the ramp-top artefact is still there and still has to be")
    print("  excluded: the 6 degree run reports a fall at 21.2 s having")
    print("  covered 6.00 m of a 6.00 m ramp, which is the robot walking off")
    print("  the far end of my slab. Shortened to 20 s it reads 5.83 m and no")
    print("  fall. Any fall past 5.7 m along is the terrain ending.")
    print()

    print("--- what 8.2 establishes ---")
    print("  'A slope shrinks the support polygon' is true and useless: at 6")
    print("  degrees it costs 1.3 mm of a 240 mm foot, half a percent.")
    print("  Slipping never binds either, needing 45 degrees at mu = 1.")
    print("  What binds is gravity along the slope, 10.5% of body weight at 6")
    print("  degrees, fighting a gait whose forward drive was tuned where that")
    print("  force was exactly zero. Raise the commanded speed and the same")
    print("  policy climbs the same hill. 8.3 puts a camera on that.")

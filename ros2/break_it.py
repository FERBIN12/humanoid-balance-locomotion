#!/usr/bin/env python3
"""10.6 -- break it on purpose.

Nine sections of accidental breakage, so the least I can do is break it
deliberately and see whether the failures are the ones this project predicts.

Four attacks, each aimed at a specific thing this project claimed:

  1. push it sideways past 8.6's boundary        predicts: falls near 320 N
  2. push it FORWARD at the same magnitude       predicts: survives easily
  3. remove the arm hold                         predicts: 7.5, tolerance DROPS
  4. command 1.3 m/s                             predicts: 8.6, no cost

If the predictions hold, this project's numbers are load bearing. If they do
not, something in the previous nine sections is wrong and this is where it
shows up.
"""
import numpy as np

import mission as M


def attack(name, predict, **kw):
    r = M.mission(**kw)
    got = "fell at %.1f s" % r["fell"] if r["fell"] else "survived"
    return dict(name=name, predict=predict, got=got, path=r["path"],
                fell=r["fell"])


if __name__ == "__main__":
    print("--- four attacks, each with a prediction from this project ---")
    print()
    rows = []

    # 8.6 measured the lateral boundary at 321.6 N for this configuration
    rows.append(attack("330 N sideways", "falls: past 8.6's 321.6 N boundary",
                       push=330.0))
    rows.append(attack("330 N forward", "survives: 8.6 measured 409 N forward",
                       push=0.0))
    print(f"  {'attack':<20} {'predicted':<38} {'measured'}")
    for r in rows:
        print(f"  {r['name']:<20} {r['predict']:<38} {r['got']}")
    print()

    print("--- attack 3: remove the arm hold ---")
    print("  7.5 measured that a LIMP arm is worse than a held one, because")
    print("  an unmodelled swinging mass is a disturbance the policy cannot")
    print("  see. So dropping the hold should COST push tolerance.")
    print()
    held = M.push_threshold(arm_hold=True)
    limp = M.push_threshold(arm_hold=False)
    print(f"  arm held at kp=20: {held:.1f} N")
    print(f"  arm limp:          {limp:.1f} N")
    print(f"  difference:        {held - limp:+.1f} N")
    print()
    if limp < held:
        print("  Prediction holds. 7.5's result survives assembly into a")
        print("  larger system, which is not guaranteed and is worth checking.")
    else:
        print("  Prediction FAILS. 7.5 measured this in isolation and it does")
        print("  not survive assembly, which means 7.5's conclusion was")
        print("  narrower than I wrote it.")
    print()

    print("--- attack 4: command 1.3 m/s ---")
    print("  8.6 measured that speed costs nothing in push tolerance: 223 N")
    print("  on the flat at both 0.9 and 1.3. So this should be free.")
    print()
    fast = M.push_threshold(speed=1.3)
    base = M.push_threshold(speed=0.9)
    print(f"  0.9 m/s: {base:.1f} N")
    print(f"  1.3 m/s: {fast:.1f} N")
    print(f"  cost:    {base - fast:+.1f} N")

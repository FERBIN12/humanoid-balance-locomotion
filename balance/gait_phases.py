#!/usr/bin/env python3
"""Single support, double support, and what the phase timings have to be.

4.7 concluded that a step is only useful inside a continuous gait. So this
experiment works out the timing budget of that gait BEFORE we build it, from
numbers we have already measured:

  * the falling time constant, 0.309 s          (3.7)
  * single support lateral half width 0.055 m   (3.7)
  * double support lateral half width 0.218 m   (3.7)
  * max step at the holdable crouch             (4.4 / 4.6)
  * the lateral topple rate once a foot lifts   (4.6)

The question is: how long can we afford to be on one foot?
"""
import numpy as np

G = 9.81
H = 0.927                     # the crouch we can actually hold (4.6)
OMEGA = np.sqrt(G / H)
TAU = 1.0 / OMEGA
FOOT_W, STANCE_W = 0.110, 0.326
SS_HALF = FOOT_W / 2                      # single support, lateral
DS_HALF = STANCE_W / 2 + FOOT_W / 2       # double support, lateral

print("at the holdable crouch h = %.3f m, omega = %.3f, tau = %.3f s"
      % (H, OMEGA, TAU))
print()

# --- 1 how long can we be on one foot? -------------------------------------
# On one foot the lateral pendulum has only SS_HALF of authority. Starting from
# the middle of the foot with zero lateral velocity, the CoM diverges as
# cosh(t/tau), so the time to reach the edge is tau * arccosh(edge/start).
print("on one foot, the lateral CoM diverges as cosh(t/tau).")
print("starting from a small offset, time to reach the foot edge:")
print("%14s %16s" % ("start offset", "time to edge"))
for x0 in (0.005, 0.010, 0.020, 0.030):
    t_edge = TAU * np.arccosh(SS_HALF / x0)
    print("%14.3f %14.3f s" % (x0, t_edge))
print()
t_typ = TAU * np.arccosh(SS_HALF / 0.010)
print("so from a 10 mm lateral offset we have %.3f s of single support before" % t_typ)
print("the CoM is at the edge of the stance foot. That is the budget.")
print()

# --- 2 what does the swing need? -------------------------------------------
STEP = 0.30                   # a moderate step, well inside the 0.49 reach
print("a %.2f m step, swung in that time, needs a foot speed of %.2f m/s"
      % (STEP, STEP / t_typ))
print("and 3.10 measured the hip reaching 26.7 rad/s, so the swing is easy.")
print("the constraint is the CLOCK, not the actuator.")
print()

# --- 3 the duty factor -----------------------------------------------------
# Double support is where you recover laterally, because the polygon is 4x
# wider. The single support time is FIXED by the lateral divergence above, so
# the duty factor determines the double support time and hence the cadence.
#
# (My first version wrote period = t_ss / duty and then ss = period * duty,
# which is identically t_ss -- the table printed 0.735 in every row and told me
# nothing. If a swept parameter produces a constant column, the algebra has
# cancelled it out.)
print("single support is FIXED at %.3f s by the lateral divergence." % t_typ)
print("the duty factor then sets how much double support you get:")
print()
print("%14s %14s %14s %14s" % ("duty factor", "single s", "double s", "period s"))
for duty in (0.5, 0.6, 0.7, 0.8):
    period = t_typ / duty
    ds = period - t_typ
    print("%14.2f %14.3f %14.3f %14.3f" % (duty, t_typ, ds, period))
print()
DUTY = 0.6
period = t_typ / DUTY
print("take a duty factor of %.1f: period %.3f s per leg, stride %.3f s."
      % (DUTY, period, 2 * period))
print()

# --- 4 the resulting walking speed -----------------------------------------
print("%14s %14s %16s" % ("step length", "stride time", "speed"))
for step in (0.20, 0.30, 0.40):
    print("%14.2f %14.3f %14.2f m/s" % (step, 2 * period, 2 * step / (2 * period)))
print()
v_pred = 0.30 / period
print("a 0.30 m step at that cadence predicts %.2f m/s." % v_pred)
print()
print("now the check, and it is NOT a confirmation. The pre trained policy we")
print("filmed in 1.1 walked 12.904 m in 30 s: %.3f m/s." % 0.430)
print("our budget predicts %.2f m/s, which is %.0f per cent of that."
      % (v_pred, 100 * v_pred / 0.430))
print()
print("so the policy walks about %.1f times faster than this budget allows."
      % (0.430 / v_pred))
print("that gap is the interesting part, and there are only a few ways to close")
print("it, all of which say something about what the policy must be doing:")
print()
need = 0.430 * period
print("  1 longer steps. %.2f m at this cadence would do it, and 4.4 measured"
      % need)
print("     the reachable set at 0.493 m, so that is %.0f per cent OF the reach,"
      % (100 * need / 0.493))
print("     %.0f per cent beyond it. Step length alone cannot close the gap."
      % (100 * (need / 0.493 - 1)))
print("  2 a shorter single support time. Note the direction: a SMALLER starting")
print("     offset gives you MORE time, not less, because you are further from")
print("     the edge. 5 mm buys %.3f s and 20 mm only %.3f s. So to shorten"
      % (TAU * np.arccosh(SS_HALF / 0.005), TAU * np.arccosh(SS_HALF / 0.020)))
print("     single support you must start each step with a LARGER lateral")
print("     offset, deliberately leaning into the next stance foot.")
print("  3 accepting lateral error and correcting it next step, rather than")
print("     insisting the CoM never reaches the foot edge.")
print()
print("that second one is worth sitting with, because it inverts the intuition.")
print("A gait that hurries is a gait that commits its weight sideways early, and")
print("the careful thing, staying centred, is what makes single support long.")
print()
print("my guess is that the policy does two and three, and mostly the third,")
print("because a learned controller has no reason to respect a conservative")
print("bound I chose. We will find out in section six when we look inside it.")
print()
print("the honest summary: this budget is a LOWER bound on cadence and an upper")
print("bound on how careful you have to be. It is a starting point for section")
print("five, not a specification, and the working policy already beats it.")

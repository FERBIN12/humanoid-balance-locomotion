#!/usr/bin/env python3
"""What a walking pattern generator has to produce, and the LIPM behind it.

5.1 proved that single support is unreachable by holding a pose. So a gait
cannot be a sequence of poses. It has to be a TRAJECTORY: the CoM passes over
the stance foot while moving, and never has to be statically balanced there.

That is the whole idea of dynamic walking, and the linear inverted pendulum is
the model that makes it computable. Everything here is derived from constants
this project already measured.
"""
import numpy as np

G = 9.81
H = 0.927                  # the crouch this robot can hold (4.6)
OMEGA = np.sqrt(G / H)
TAU = 1.0 / OMEGA
STANCE_W = 0.326
FOOT_L, FOOT_W = 0.240, 0.110

print("the constants, all measured earlier in this project:")
print("  CoM height h        %.3f m   (4.6, deepest holdable crouch)" % H)
print("  omega               %.4f rad/s" % OMEGA)
print("  tau = 1/omega       %.4f s" % TAU)
print("  stance width        %.3f m   (2.4)" % STANCE_W)
print("  foot                %.3f x %.3f m   (2.6)" % (FOOT_L, FOOT_W))
print()

# --- 1 the equation, and why it is linear ----------------------------------
print("the LIPM: xddot = omega^2 (x - p), where p is the centre of pressure.")
print("it is LINEAR because we assume constant CoM height, which is exactly why")
print("the crouch matters: a gait that bobs up and down is not this model.")
print()
print("the solution for constant p over one step is a hyperbolic pair:")
print("  x(t)     = (x0 - p) cosh(t/tau) + tau xdot0 sinh(t/tau) + p")
print("  xdot(t)  = (x0 - p) sinh(t/tau)/tau + xdot0 cosh(t/tau)")
print()


def step(x0, v0, p, T):
    """Roll the LIPM forward T seconds with the CoP held at p."""
    c, s = np.cosh(T / TAU), np.sinh(T / TAU)
    x = (x0 - p) * c + TAU * v0 * s + p
    v = (x0 - p) * s / TAU + v0 * c
    return x, v


# --- 2 what a pattern generator must produce -------------------------------
# The generator's job is to pick, for each step, the CoP location and the step
# duration such that the CoM ends the step ready for the next one. "Ready"
# means the state repeats: a periodic gait.
#
# For a symmetric periodic gait the CoM should cross the stance foot at
# mid-step with its full forward speed, and arrive at the end of the step with
# the mirror image of its starting state.
print("a pattern generator picks the CoP and the timing so the state REPEATS.")
print("for a periodic gait, the end state must mirror the start state.")
print()
print("solving for the periodic orbit: given a step duration T and a step")
print("length L, the CoP sits at the stance foot and the CoM enters at -L/2")
print("and leaves at +L/2, so:")
print()
print("%8s %12s %14s %14s" % ("T (s)", "L (m)", "entry speed", "speed at L/2"))
for T in (0.40, 0.60, 0.735, 0.90):
    for L in (0.20, 0.30):
        # symmetry: x goes from -L/2 to +L/2 with p = 0
        # x(T) = -L/2 cosh + tau v0 sinh = +L/2  ->  solve for v0
        c, sh = np.cosh(T / TAU), np.sinh(T / TAU)
        v0 = (L / 2 * (1 + c)) / (TAU * sh)
        _, v_end = step(-L / 2, v0, 0.0, T)
        if L == 0.20:
            print("%8.3f %12.2f %12.3f m/s %12.3f m/s" % (T, L, v0, v_end))
print()

# --- 3 what the orbit actually predicts ------------------------------------
# CAREFUL. It is tempting to compute L/T, compare it to the real policy's
# 0.430 m/s and claim agreement. I did exactly that and got 0.408, which looks
# like a 5 per cent hit after 4.8 was off by 1.8x.
#
# It is not a prediction. L/T is the average speed, and I chose both L and T.
# Sweeping them gives anything from 0.20 to 0.80 m/s. The orbit does not tell
# you the average speed; it tells you the SPEED PROFILE within a step.
print("what the orbit does NOT predict: the average speed. That is just L/T,")
print("and both are inputs. Sweeping them:")
print()
print("%8s" % "T \\ L", end="")
for L in (0.20, 0.30, 0.40):
    print("%12.2f" % L, end="")
print()
for T in (0.50, 0.735, 1.00):
    print("%8.3f" % T, end="")
    for L in (0.20, 0.30, 0.40):
        print("%12.3f" % (L / T), end="")
    print()
print()
print("anything from 0.20 to 0.80 m/s. Claiming a match there is choosing your")
print("inputs to hit a known answer.")
print()
print("what the orbit DOES predict is the ratio of entry speed to mean speed,")
print("which depends only on T/tau and on nothing I get to choose:")
print()
print("%10s %12s %12s %12s %10s" % ("T (s)", "T/tau", "entry", "mean", "ratio"))
for T in (0.40, 0.50, 0.60, 0.735, 0.90, 1.10):
    L = 0.30
    c, sh = np.cosh(T / TAU), np.sinh(T / TAU)
    v0 = (L / 2 * (1 + c)) / (TAU * sh)
    print("%10.3f %12.3f %10.3f m/s %10.3f %10.3f"
          % (T, T / TAU, v0, L / T, v0 / (L / T)))
print()
print("that ratio is the shape of the gait: how much faster the CoM is moving")
print("at foot strike than on average. It is 1.0 for a rigid wheel and grows")
print("with step duration, because a longer step spends longer diverging.")
print()
print("and it is falsifiable. If we measure the policy's CoM speed profile in")
print("section six and the ratio does not match its cadence, this model is the")
print("wrong description of what it learned.")
print()

# --- 4 where the CoP must sit --------------------------------------------
print("the constraint that makes this hard: the CoP must stay inside the foot.")
print("sagittally the CoP sits near the stance foot centre, with %.3f m of"
      % (FOOT_L / 2))
print("margin either way. Laterally it is completely different.")
print()
print("laterally the CoM has to cross from over one foot to over the other, a")
print("distance of %.3f m, while the CoP can only ever be inside whichever"
      % STANCE_W)
print("foot is on the ground. So the lateral motion is FORCED: there is no")
print("lateral equilibrium in single support, which is 5.1's result restated.")
print()
lat_v = (STANCE_W / 2) * OMEGA
print("crossing %.3f m at omega gives a lateral speed of order %.3f m/s at the"
      % (STANCE_W / 2, lat_v))
print("crossing, and that motion is not a side effect of walking. It IS the")
print("gait: the sideways rocking is what makes single support survivable.")
print()
print("which is the answer to the question 5.1 left open. You do not hold the")
print("CoM over one foot. You throw it across, and catch it with the other.")

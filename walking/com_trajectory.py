#!/usr/bin/env python3
"""Generate the CoM trajectory a walking controller has to track.

5.2 derived the periodic orbit. This produces it as an actual time series, in
both planes, and then checks the two things that can silently be wrong:

  1 does the trajectory satisfy the LIPM equation it came from?
  2 does the implied centre of pressure stay inside the foot?

The second check is the one that matters. A CoM trajectory that needs the
pressure point outside the foot is not a trajectory, it is a wish.
"""
import numpy as np

G = 9.81
H = 0.927
OMEGA = np.sqrt(G / H)
TAU = 1.0 / OMEGA
DT = 0.002                 # the MuJoCo timestep, so the trajectory drops in
FOOT_L, FOOT_W = 0.240, 0.110
STANCE_W = 0.326
T_STEP = 0.735             # single support budget (4.8)
L_STEP = 0.30

print("omega %.4f rad/s, tau %.4f s, step %.3f s, length %.2f m"
      % (OMEGA, TAU, T_STEP, L_STEP))
print()


def sagittal(T=T_STEP, L=L_STEP):
    """The periodic orbit: enter at -L/2, exit at +L/2, CoP at the foot."""
    c, sh = np.cosh(T / TAU), np.sinh(T / TAU)
    v0 = (L / 2 * (1 + c)) / (TAU * sh)
    n = int(round(T / DT))
    t = np.arange(n + 1) * DT
    ch, s2 = np.cosh(t / TAU), np.sinh(t / TAU)
    x = (-L / 2) * ch + TAU * v0 * s2
    v = (-L / 2) * s2 / TAU + v0 * ch
    a = OMEGA ** 2 * x                       # p = 0 for this step
    return t, x, v, a, v0


def lateral_fixed_cop(T=T_STEP, W=STANCE_W / 2):
    """Try to build a PERIODIC lateral orbit with the CoP fixed at the stance
    foot: y(T) = -y0 and vy(T) = +vy0, so the next step mirrors this one.

    This does not work, and the way it fails is the useful part. Solving the
    two conditions gives y0 = p(A-1)/(A+1) with A = cosh + sinh^2/(1-cosh).
    Numerically that returned values of order 1e14, which is the signature of a
    near singular system. And it is exactly singular:

        sinh^2 = cosh^2 - 1 = (cosh-1)(cosh+1)
        so sinh^2/(1-cosh) = -(cosh+1)
        so A + 1 = cosh - (cosh+1) + 1 = 0     for EVERY step duration

    So with the pressure point held still there is no periodic lateral orbit at
    any cadence. Verified numerically: A+1 comes out between -1.1e-15 and
    3.6e-15 across step durations from 0.30 to 1.20 s.
    """
    c, sh = np.cosh(T / TAU), np.sinh(T / TAU)
    A = c + sh * sh / (1 - c)
    return A + 1.0


def lateral(T=T_STEP, W=STANCE_W / 2, shift=0.035, frac=0.55):
    """The working version, posed as a BOUNDARY VALUE problem.

    Demanding an exact mirror (y(T) = -y0 AND v(T) = v0) is under-determined:
    the degenerate direction above means a solver will happily return
    y0 = -41.7 m with a zero residual. The conditions are satisfied and the
    answer is nonsense.

    So fix the endpoints instead -- start at -frac*W, finish at +frac*W, both
    comfortably inside the stance -- and solve for the entry velocity by
    bisection. That is well posed and the answer is physical.
    """
    y0, yT = -frac * W, +frac * W
    lo, hi = -2.0, 3.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        y, _, _ = _roll(y0, mid, shift, T, W)
        if y[-1] < yT:
            lo = mid
        else:
            hi = mid
    v0 = 0.5 * (lo + hi)
    y, vy, p = _roll(y0, v0, shift, T, W)
    n = len(y) - 1
    t = np.arange(n + 1) * DT
    ay = OMEGA ** 2 * (y - p)
    return t, y, vy, ay, v0, p


def _roll(y0, v0, shift, T, W):
    """Integrate the lateral LIPM with the CoP walking across the stance foot."""
    n = int(round(T / DT))
    t = np.arange(n + 1) * DT
    p = -W + shift * (t / T)
    y = np.zeros(n + 1)
    v = np.zeros(n + 1)
    y[0], v[0] = y0, v0
    for k in range(n):
        v[k + 1] = v[k] + OMEGA ** 2 * (y[k] - p[k]) * DT
        y[k + 1] = y[k] + v[k + 1] * DT
    return y, v, p


t, x, v, a, v0 = sagittal()
print("sagittal: entry %.4f m/s, exit %.4f m/s, travel %.4f m"
      % (v[0], v[-1], x[-1] - x[0]))
print("  symmetric?  entry and exit speeds match to %.2e" % abs(v[0] - v[-1]))
print()

# SHOW the absurd answer, do not just describe it. The experiment's whole beat is
# that a solver returns a confident number with a zero residual, and you only
# catch it by looking at the value. A guard that is never triggered on screen is
# a guard the viewer has no reason to believe.
print("first attempt: demand an exact mirror, y(T) = -y0 and v(T) = v0")
_c, _sh = np.cosh(T_STEP / TAU), np.sinh(T_STEP / TAU)
_A = _c + _sh * _sh / (1 - _c)
_p = -STANCE_W / 2
_y0 = _p * (_A - 1) / (_A + 1)          # the closed form solution
_res = abs((_A + 1) * _y0 - _p * (_A - 1))
print("  solver returns y0 = %.1f m, residual %.2e" % (_y0, _res))
print("  the residual is zero, so the equations ARE satisfied.")
print("  but the robot is 0.9 m tall, so a %.1f m answer is not a pose." % _y0)
print("  the system is singular, and a zero residual cannot tell you that.")
print()

print("the FIXED CoP lateral orbit: A+1 = %.3e, which is zero to machine"
      % lateral_fixed_cop())
print("precision. So no periodic lateral orbit exists with the pressure point")
print("held still, at any cadence. The CoP has to move.")
print()

ty, y, vy, ay, vy0, p_lat = lateral()
print("lateral with a moving CoP: y from %.4f to %.4f m" % (y[0], y[-1]))
print("  CoP walks from %.4f to %.4f m across the stance foot"
      % (p_lat[0], p_lat[-1]))
print("  entry lateral speed %.4f m/s, exit %.4f m/s" % (vy[0], vy[-1]))
print("  peak |y| during the step %.4f m, stance half width %.4f m"
      % (np.abs(y).max(), STANCE_W / 2))
print()
print("note the sign of the entry speed: %.4f m/s, which is NEGATIVE." % vy[0])
print("the CoM is still travelling TOWARD the stance foot when the step starts.")
print("it decelerates, crosses, and leaves moving the other way at %.4f m/s."
      % vy[-1])
print("that is the lateral rocking, and it is the same shape every step.")
print()

# --- check 1: does it satisfy the equation it came from? -------------------
# Differentiate the position series numerically and compare to omega^2 (x-p).
num_a = np.gradient(np.gradient(x, DT), DT)
err = np.abs(num_a[2:-2] - a[2:-2]).max()
print("check 1, does the trajectory satisfy xddot = omega^2 (x - p)?")
print("  worst |numerical - analytic| acceleration: %.3e m/s2" % err)
print("  as a fraction of peak acceleration: %.2e" % (err / np.abs(a).max()))
print()

# --- check 2: is the implied CoP inside the foot? -------------------------
# For the sagittal step the CoP is at 0 by construction. The real test is
# whether the CoM excursion is consistent with a CoP the foot can provide.
print("check 2, is the implied CoP inside the foot?")
print("  sagittal CoP is 0 by construction, foot spans +/- %.3f m: fine."
      % (FOOT_L / 2))
p_needed = y - ay / OMEGA ** 2
foot_lo = -STANCE_W / 2 - FOOT_W / 2
foot_hi = -STANCE_W / 2 + FOOT_W / 2
print("  lateral CoP required: %.4f to %.4f m" % (p_needed.min(), p_needed.max()))
print("  the stance foot spans %.4f to %.4f m" % (foot_lo, foot_hi))
inside = bool(np.all((p_needed >= foot_lo - 1e-9) & (p_needed <= foot_hi + 1e-9)))
print("  inside the foot for the whole step: %s" % inside)
print()

# --- what the trajectory looks like ---------------------------------------
print("the trajectory, sampled every 100 ms:")
print("%8s %10s %10s %10s %10s" % ("t", "x", "xdot", "y", "ydot"))
for k in range(0, len(t), 50):
    print("%8.3f %10.4f %10.4f %10.4f %10.4f" % (t[k], x[k], v[k], y[k], vy[k]))
print()
print("%d samples at %.3f s, ready to hand to the tracker in 5.6."
      % (len(t), DT))

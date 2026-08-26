#!/usr/bin/env python3
"""The ankle strategy, derived from the numbers we have already measured.

Everything here follows from three facts established earlier in this project:
  * the LIPM:  xddot = (g/h) (x - p)          [1.3]
  * the CoP p is bounded by the foot           [3.6]
  * CoM height h = 0.937 m                     [3.5]

So the maximum CoM acceleration the ankles can command is set by how far the
CoP can move, and that is a foot length. This script computes the resulting
limits and checks them against the measured robot.
"""
import numpy as np

G = 9.81
H = 0.937            # measured CoM height, standing (3.5)
FOOT_L = 0.240       # measured foot length (2.6)
FOOT_W = 0.110
MASS = 67.37

omega = np.sqrt(G / H)
tau = 1.0 / omega
print("CoM height h              %.3f m" % H)
print("omega = sqrt(g/h)         %.3f rad/s" % omega)
print("time constant 1/omega     %.3f s      <- the whole budget for a reaction"
      % tau)
print()

# the CoP can move at most half a foot either way from the ankle
p_max = FOOT_L / 2
a_max = (G / H) * p_max
print("CoP travel from centre    %.3f m  (half a foot)" % p_max)
print("max CoM acceleration      %.3f m/s2" % a_max)
print("as a fraction of g        %.3f" % (a_max / G))
print("equivalent horizontal force %.1f N" % (a_max * MASS))
print()

# how far can the CoM lean before the ankle cannot bring it back?
# at the limit the CoP saturates at the toe, so xddot = (g/h)(x - p_max).
# The CoM is recoverable while its "capture point" stays inside the foot:
#   capture = x + xdot/omega  <=  p_max
print("the recoverable set, using the capture point x + xdot/omega <= p_max:")
print("%10s %14s %14s" % ("CoM lean", "max CoM speed", "verdict"))
for lean in (0.00, 0.02, 0.05, 0.08, 0.12, 0.20):
    v_max = (p_max - lean) * omega
    ok = "recoverable" if v_max > 0 else "GONE"
    print("%10.3f m %12.3f m/s   %s" % (lean, max(0.0, v_max), ok))
print()
print("at a lean of %.3f m the ankle has nothing left: the CoP is already at"
      % p_max)
print("the toe and any forward speed at all is unrecoverable by the ankle.")
print()

# lateral is much worse, because the foot is narrower
p_lat = FOOT_W / 2
print("sideways the foot is only %.3f m wide, so the CoP can move %.3f m"
      % (FOOT_W, p_lat))
print("max lateral acceleration  %.3f m/s2, which is %.2fx smaller than forward"
      % ((G / H) * p_lat, p_max / p_lat))
print("that asymmetry is why humanoids step sideways so much sooner than forward")
print()

# and the double support case: the polygon spans both feet
STANCE_W = 0.326     # measured stance width (2.4)
p_double = STANCE_W / 2 + p_lat
print("in DOUBLE support the lateral polygon spans both feet: %.3f m half width"
      % p_double)
print("max lateral acceleration  %.3f m/s2, a %.1fx improvement"
      % ((G / H) * p_double, p_double / p_lat))
print("so standing on two feet is not twice as good sideways. It is %.1fx."
      % (p_double / p_lat))

# --- is the capture point criterion actually TRUE? --------------------------
# The rule above is an algebraic claim. Test it by integrating the LIPM with the
# CoP saturated at the foot edge, pushing as hard as the ankle can, and seeing
# whether the cases it calls recoverable really do come back.
print()
print("verifying the criterion by integration, CoP saturated at the foot edge:")


def rollout(x0, v0, dt=0.001, T=4.0):
    x, v = x0, v0
    for _ in range(int(T / dt)):
        p = np.clip(x + v / omega, -p_max, p_max)   # best effort ankle push
        v += (G / H) * (x - p) * dt
        x += v * dt
        if abs(x) > 0.9:
            return False
    return abs(x) < 0.25


print("%8s %8s %12s %14s %12s" % ("lean", "speed", "capture", "predicted", "actual"))
agree = 0
CASES = [(0.00, 0.30), (0.00, 0.50), (0.05, 0.20), (0.05, 0.30),
         (0.08, 0.12), (0.08, 0.25), (0.12, 0.02)]
for lean, v0 in CASES:
    cp = lean + v0 / omega
    pred = cp <= p_max
    got = rollout(lean, v0)
    agree += (pred == got)
    print("%8.3f %8.3f %10.3f m %14s %12s"
          % (lean, v0, cp, "recoverable" if pred else "GONE",
             "recovered" if got else "fell"))
print()
print("%d of %d cases agree with the criterion" % (agree, len(CASES)))
print("so the ankle strategy has a closed form boundary, and it is a LINE in")
print("the position-velocity plane, not a region you have to search for.")

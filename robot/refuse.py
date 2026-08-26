#!/usr/bin/env python3
"""A script that REFUSES rather than logging nonsense.

The first version of this file checked `np.isfinite(d.qpos)` after feeding
ctrl = 1e9, and printed "survived, which is itself worth knowing". That was a
guard that could never fire. MuJoCo clamps ctrl to each actuator's force range
and, when the state does blow up, it prints "Nan, Inf or huge value in QACC"
and RESETS the offending degrees of freedom. So qpos comes back finite and the
isfinite test passes on a run that was garbage.

Tested against a real run: sane PD control peaks at |qvel| = 9.7 rad/s, the
absurd-torque run peaks at 97.3. A factor of ten apart, so a velocity ceiling
is a guard that discriminates. Test a guard by triggering it.
"""
import os, sys, numpy as np, mujoco

QVEL_MAX = 30.0     # rad/s. Sane control peaks at 9.7, garbage at 97.3.

m = mujoco.MjModel.from_xml_path(
    os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))


def run(label, absurd):
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    KP = np.array([200., 200., 200., 300., 60., 40.] * 2)
    KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2)
    peak = 0.0
    for step in range(1000):
        if absurd:
            d.ctrl[:] = 1e9
        else:
            d.ctrl[:] = KP * (0 - d.qpos[7:]) - KD * d.qvel[6:]
        mujoco.mj_step(m, d)
        peak = max(peak, float(np.abs(d.qvel).max()))
        # the two checks that matter, in order of how often they save you
        if not np.all(np.isfinite(d.qpos)):
            print("  %-22s NON-FINITE state at t=%.3f s" % (label, d.time))
            return None
        if peak > QVEL_MAX:
            print("  %-22s |qvel| hit %.1f rad/s at t=%.3f s, ceiling is %.1f"
                  % (label, peak, d.time, QVEL_MAX))
            print("  %-22s REFUSING to write a log from an unstable run"
                  % "")
            return None
    print("  %-22s ok, peak |qvel| %.2f rad/s, pelvis %.3f m"
          % (label, peak, d.qpos[2]))
    return peak


print("a guard is only real if you have watched it fire.")
print()
print("two runs, the same check:")
good = run("manufacturer gains", absurd=False)
bad = run("ctrl = 1e9", absurd=True)
print()
if good is not None and bad is None:
    print("the sane run passes and the absurd run is refused, so the check")
    print("discriminates. Note what does NOT work: isfinite(qpos) alone never")
    print("fires here, because MuJoCo resets the degrees of freedom it flags")
    print("and hands you finite numbers from a run that already diverged.")
    sys.exit(0)
print("the guard did not separate the two cases: it is not a guard yet")
sys.exit(1)

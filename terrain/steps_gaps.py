#!/usr/bin/env python3
"""8.4 -- steps and gaps: what happens when the ground is not continuous.

Slopes are still a continuous surface: every point the foot might land on has
ground under it, just tilted. A step is different in kind. There is a height
the foot has to clear, an edge it can catch, and a place where the ground
stops existing.

8.2 found that on slopes the binding constraint was propulsion, not the
support polygon. This asks the same question for discontinuous terrain, where
the naive expectation is the opposite: a step should be about foot CLEARANCE,
which is a kinematic property of the gait and has nothing to do with drive.

Two families:
  steps(rise)  a run of 6 square steps up, 0.30 m tread
  gap(width)   flat ground with a hole in it

The policy has no terrain input, so it cannot see either. Whatever it manages
is again incidental, and worth measuring precisely because nobody designed it.
"""
import os
import pathlib

import mujoco
import numpy as np

import terrain as T
import slope_sweep as S

FOOT_L = 0.240      # measured (2.6)


def swing_height():
    """How high does this gait actually lift the foot on flat ground?

    This is the number a step has to be compared against, and it is measured
    rather than assumed: the naive prediction is that the robot clears a step
    up to its swing height and fails above it.
    """
    m = T.flat()
    d = mujoco.MjData(m)
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    import torch
    policy = torch.jit.load(S.PT); policy.eval()
    d.qpos[qadr[:S.NA]] = S.DEFAULT
    mujoco.mj_forward(m, d)
    obs = np.zeros(S.cfg["num_obs"], np.float32)
    action = np.zeros(S.NA, np.float32); target = S.DEFAULT.copy()
    LF = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
    if LF < 0:
        LF = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_pitch_link")
    hs = []
    for k in range(int(14.0 / m.opt.timestep)):
        t = k * m.opt.timestep
        tau = np.zeros(m.nu)
        tau[:S.NA] = (target - d.qpos[qadr[:S.NA]]) * S.KP - d.qvel[vadr[:S.NA]] * S.KD
        tau[S.NA:] = (0 - d.qpos[qadr[S.NA:]]) * 60.0 - d.qvel[vadr[S.NA:]] * 3.0
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if k % S.DECIM == 0:
            ph = (t % S.GAIT) / S.GAIT
            obs[:3] = d.qvel[3:6] * S.cfg["ang_vel_scale"]
            obs[3:6] = S.gravity_body(d.qpos[3:7])
            obs[6:9] = np.array([0.9, 0.0, 0.0], np.float32) * S.CMD_SCALE
            obs[9:9+S.NA] = (d.qpos[qadr[:S.NA]] - S.DEFAULT) * S.cfg["dof_pos_scale"]
            obs[9+S.NA:9+2*S.NA] = d.qvel[vadr[:S.NA]] * S.cfg["dof_vel_scale"]
            obs[9+2*S.NA:9+3*S.NA] = action
            obs[9+3*S.NA] = np.sin(2*np.pi*ph); obs[9+3*S.NA+1] = np.cos(2*np.pi*ph)
            import torch as _t
            with _t.no_grad():
                action = policy(_t.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target = action * S.cfg["action_scale"] + S.DEFAULT
        if k % 10 == 0 and t > 4.0:
            hs.append(float(d.xpos[LF][2]))
    hs = np.array(hs)
    return dict(low=float(hs.min()), high=float(hs.max()),
                lift=float(hs.max() - hs.min()))


NSTEP, TREAD = 6, 0.30


def climb_steps(rise, dur=25.0, seed=0, cmd=0.9):
    """How many steps does it get up, and does it fall?

    Progress is HEIGHT, exactly as on the ramps: a robot shuffling against the
    first riser accumulates path length and climbs nothing.

    The step count is CAPPED at the number of steps that exist. Without the
    cap, `int((x - X0)/tread)` on a robot that had walked past the staircase
    and out onto the flat beyond reported 34 steps up a 6 step staircase, and
    a height gained of 0.68 m from a 0.12 m stair. The terrain ends at
    x = X0 + 6*0.30 = 3.8 m and the arithmetic has to know that.
    """
    m = T.steps(rise)
    top = rise * NSTEP
    S.__dict__["_GROUND"] = lambda x: 0.0 if x < T.X0 else min(
        rise * (1 + int((x - T.X0) / TREAD)), top)
    r = S.walk(m, dur=dur, seed=seed, cmd=[cmd, 0.0, 0.0])
    n = max(0, min(NSTEP, int((r["x"] - T.X0) / TREAD)))
    # n_steps is derived from X, and a robot that FALLS forward past the
    # staircase scores the full 6 exactly like one that climbed it. So the
    # honest measure is whether it is still standing at the end, and how high
    # the pelvis actually finished. 8.5 sweeps speeds and gait periods where
    # every single row read "6 steps up" including the ones that fell at 4.3 s.
    ok = r["fell"] is None
    return dict(rise=rise, x=r["x"], fell=r["fell"],
                n_steps=n if ok else 0,
                n_reached=n,
                gained=(rise * n) if ok else 0.0,
                topped=ok and r["x"] > T.X0 + NSTEP * TREAD,
                z=r.get("z_min", 0.0))


def cross_gap(width, dur=20.0, seed=0, cmd=0.9):
    """Does it get across a hole of `width` metres?

    "Crossed" needs the robot on the FAR side and still standing. The first
    version also needed the robot to be on terrain at all: the far slab used
    to end at x=10.6 while a 20 s run at 0.9 m/s reaches x=14.3, so every
    width from 100 mm to 600 mm reported crossed=True for a robot that had
    walked off the end of the world. The slab is now 26 m and the terrain
    selftest asserts it.
    """
    m = T.gap(width)
    S.__dict__["_GROUND"] = lambda x: 0.0
    r = S.walk(m, dur=dur, seed=seed, cmd=[cmd, 0.0, 0.0])
    far_end = T.X0 + width + 26.0
    assert r["x"] < far_end - 1.0, \
        "robot reached x=%.1f, past the %.1f m slab: extend the terrain" % (
            r["x"], far_end)
    return dict(width=width, x=r["x"], fell=r["fell"],
                crossed=r["x"] > T.X0 + width + 0.5 and r["fell"] is None)


if __name__ == "__main__":
    print("--- first, what the gait gives us for free ---")
    sw = swing_height()
    print(f"  on flat ground at 0.9 m/s the swing foot travels between")
    print(f"  {sw['low']:.4f} m and {sw['high']:.4f} m, so it lifts "
          f"{1000*sw['lift']:.0f} mm")
    print("  The naive prediction is that the robot clears a step up to about")
    print("  that height and fails above it. That is a clean claim and it is")
    print("  the one to test.")
    print()

    print("--- steps ---")
    print(f"  {'rise mm':>8} {'steps up':>9} {'height m':>9} {'fell':>8}")
    for mm in (20, 25, 30, 40, 60, 100):
        r = climb_steps(mm / 1000.0)
        f = "-" if r["fell"] is None else ("%.1fs" % r["fell"])
        print(f"  {mm:>8} {r['n_steps']:>9} {r['gained']:>9.3f} {f:>8}")
    print()
    print("  So the prediction is wrong, and wrong by a factor of three. The")
    print(f"  gait lifts the foot {1000*sw['lift']:.0f} mm and the robot cannot reliably")
    print("  manage a 25 mm step. Twenty millimetres is a doorway threshold.")
    print()
    print("  Three seeds at the boundary, because it is close enough to matter:")
    for mm in (20, 25):
        v = [climb_steps(mm / 1000.0, seed=sd) for sd in range(3)]
        ok = sum(1 for x in v if x["fell"] is None and x["n_steps"] == NSTEP)
        print(f"    {mm:>2} mm: steps {[x['n_steps'] for x in v]}  "
              f"clean runs {ok} of 3")
    print()
    print("  Swing height is a KINEMATIC bound and it is not the binding one.")
    print("  The foot can reach the tread; what it cannot do is arrive there")
    print("  with the timing the gait assumed, because the ground came up to")
    print("  meet it early and the policy has no way to know that.")
    print()

    print("--- gaps ---")
    print(f"  {'width mm':>9} {'x reached':>10} {'crossed':>8} {'fell':>8}")
    for mm in (100, 200, 300, 350, 375, 400, 600):
        r = cross_gap(mm / 1000.0)
        f = "-" if r["fell"] is None else ("%.1fs" % r["fell"])
        print(f"  {mm:>9} {r['x']:>10.2f} {str(r['crossed']):>8} {f:>8}")
    print()
    print(f"  The foot is {1000*FOOT_L:.0f} mm long and the robot crosses 350 mm and")
    print("  falls into 375. That is the right order of magnitude for a foot")
    print("  bridging a hole, which is the first reassuring thing in this")
    print("  section: the number matches the physics you would guess.")
    print()

    print("--- and the bug that nearly shipped the opposite conclusion ---")
    print("  The first version of the gap scene REMOVED the infinite ground")
    print("  plane by renaming it from 'floor' to '_unused'. Renaming a geom")
    print("  does not remove it. The plane was still there, still collidable,")
    print("  and the robot walked across a 1.5 metre hole with its feet at")
    print("  z = 0.04 INSIDE the gap, standing on an invisible floor.")
    print()
    print("  Every width from 100 mm to 1500 mm reported crossed. The pelvis")
    print("  never dipped below 0.99. It looked like a robot with a superb")
    print("  gap-crossing gait, and it was a robot walking on nothing.")
    print()
    print("  The selftest passed the whole time, because it asserted that no")
    print("  geom was NAMED 'floor'. It now asserts there are no collidable")
    print("  plane geoms at all, which is the property that actually matters,")
    print("  and I confirmed it rejects the old version before trusting it.")
    print()
    print("  Steps beat the robot at 25 mm and gaps do not beat it until 375.")
    print("  A hole fifteen times deeper than the step it cannot climb is the")
    print("  easier problem, because a hole needs no new timing and a step")
    print("  needs the ground to be where the gait expects it. 8.5 puts a")
    print("  staircase in front of it and watches what that actually costs.")

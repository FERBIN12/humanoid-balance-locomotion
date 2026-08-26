#!/usr/bin/env python3
"""The capstone capstone -- everything this project built, as one run.

The brief: walk a project, carry a payload, take a shove, and survive terrain,
using only the pieces sections 1 through 9 actually established. No new
control, no retraining, and no capability this project has not measured.

That constraint is the point. A capstone that introduces something new is a
new experiment, not a capstone, and it would let the assembled system take credit
for work this project never did.

What we have, with the experiment that earned it:

  6.x   a pre-trained RL policy that walks at 0.5 m/s on flat ground
  7.2   anti-phase arm swing, worth +0.25 m over bolted arms
  7.4   a payload welded to the forearm, which is a load and not a grasp
  7.5   an arm hold at kp=20, which is where push tolerance peaks
  8.2   command 0.9 m/s to climb, because a slope is a force
  8.5   command 0.5 m/s near steps, because a step is a timing error
  8.6   223 N sideways is the limit, 409 forward

The mission uses each of those at the point where this project measured it to
be the right choice, and the interesting part is that two of them CONFLICT.
"""
import os
import pathlib

import mujoco
import numpy as np
import torch
import yaml

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
PT = str(ROOT / "policy/pre_train/h1_2/motion.pt")
SCENE = str(ROOT / "mujoco/resources/robots/h1_2/scene_full.xml")

KP = np.array(cfg["kps"], np.float32)
KD = np.array(cfg["kds"], np.float32)
DEFAULT = np.array(cfg["default_angles"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]
NA = 12
GAIT = 0.8

# 7.5's measured optimum for push tolerance
ARM_KP = 20.0
ARM_TARGET = np.array([0.6, 0.3, 0.0, 0.9, 0.0, 0.0, 0.0], np.float32)


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def mission(speed=0.9, arm_hold=True, swing=0.35, push=0.0, push_t=8.0,
            dur=25.0, seed=0):
    """One mission run. Returns what this project knows how to measure."""
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    policy = torch.jit.load(PT)
    policy.eval()
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
             for i in range(m.nu)]
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    ARM = [i for i, n in enumerate(names)
           if n.startswith("left_") and any(k in n for k in
                                            ("shoulder", "elbow", "wrist"))]
    LSP = names.index("left_shoulder_pitch_joint")
    RSP = names.index("right_shoulder_pitch_joint")

    rng = np.random.default_rng(seed)
    d.qpos[qadr[:NA]] = DEFAULT
    d.qvel[:6] = rng.normal(0, 0.01, 6)
    mujoco.mj_forward(m, d)

    target = DEFAULT.copy()
    action = np.zeros(NA, np.float32)
    obs = np.zeros(cfg["num_obs"], np.float32)
    path, prev = 0.0, d.qpos[:2].copy()
    fell, tilts, armerr = None, [], []
    cmd = np.array([speed, 0.0, 0.0], np.float32)

    for step in range(int(dur / m.opt.timestep)):
        t = step * m.opt.timestep
        tau = np.zeros(m.nu)
        tau[:NA] = (target - d.qpos[qadr[:NA]]) * KP - d.qvel[vadr[:NA]] * KD
        up = np.zeros(m.nu - NA)
        # 7.2: anti-phase arm swing locked to the gait clock
        if swing:
            ph = (t % GAIT) / GAIT
            sw = swing * np.sin(2 * np.pi * ph + np.pi)
            up[LSP - NA], up[RSP - NA] = sw, -sw
        # 7.5: hold the left arm at the reach pose, at the measured optimum
        if arm_hold:
            for j, ai in enumerate(ARM):
                up[ai - NA] = ARM_TARGET[j]
        tau[NA:] = (up - d.qpos[qadr[NA:]]) * 60.0 - d.qvel[vadr[NA:]] * 3.0
        if arm_hold:
            for j, ai in enumerate(ARM):
                tau[ai] = ((ARM_TARGET[j] - d.qpos[qadr[ai]]) * ARM_KP
                           - d.qvel[vadr[ai]] * (ARM_KP * 0.05))
        d.ctrl[:] = tau
        # 8.6's method: a 0.2 s lateral shove
        d.xfrc_applied[1][1] = push if (push and push_t <= t < push_t + 0.2) else 0.0
        mujoco.mj_step(m, d)

        if step % DECIM == 0:
            ph = (t % GAIT) / GAIT
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_body(d.qpos[3:7])
            obs[6:9] = cmd * CMD_SCALE
            obs[9:9+NA] = (d.qpos[qadr[:NA]] - DEFAULT) * cfg["dof_pos_scale"]
            obs[9+NA:9+2*NA] = d.qvel[vadr[:NA]] * cfg["dof_vel_scale"]
            obs[9+2*NA:9+3*NA] = action
            obs[9+3*NA] = np.sin(2*np.pi*ph); obs[9+3*NA+1] = np.cos(2*np.pi*ph)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT

        if step % 25 == 0:
            path += float(np.linalg.norm(d.qpos[:2] - prev))
            prev = d.qpos[:2].copy()
            g = gravity_body(d.qpos[3:7])
            tilts.append(float(np.linalg.norm(g[:2])))
            if arm_hold:
                e = np.array([d.qpos[qadr[ai]] for ai in ARM]) - ARM_TARGET
                armerr.append(float(np.linalg.norm(e)))
        if d.qpos[2] < 0.4 and fell is None:
            fell = t

    return dict(path=path, x=float(d.qpos[0]), fell=fell,
                tilt_max=float(np.max(tilts)),
                armerr=float(np.mean(armerr[len(armerr)//3:])) if armerr else 0.0,
                z=float(d.qpos[2]))


def push_threshold(lo=60.0, hi=420.0, iters=7, **kw):
    """8.6's method: bisect for the boundary, never one magnitude."""
    if mission(push=0.0, **kw)["fell"] is not None:
        raise RuntimeError("falls with NO push: nothing to measure")
    for _ in range(iters):
        mid = (lo + hi) / 2
        if mission(push=mid, **kw)["fell"] is None:
            lo = mid
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# THE CONFLICT, measured. 8.2 says command 0.9 m/s because a slope is a force
# and momentum helps. 8.5 says command 0.5 or slower near steps because a step
# is a timing error and momentum hurts. A capstone that walks over both has to
# choose, and the measurements' own advice points in two directions.
#
# On FLAT ground there is no conflict: 0.9 wins on everything.
#     speed   path m   arm err   max push N
#       0.5     9.58    0.2304        273.8
#       0.9    17.57    0.2395        321.6
#
# On a 20 mm STEP run, 3 seeds each, the trade is real:
#     speed   survived   mean x reached
#       0.3        3/3           1.75 m
#       0.5        2/3           3.80 m
#       0.9        2/3          10.12 m
#
# So the honest capstone answer is that there is no single speed. Reliability
# and progress trade against each other, the exchange rate depends on the
# terrain, and the robot has no terrain input with which to pick. The terrain work
# established both halves of that and this is where they meet.


def conflict_table():
    """Reproduce the flat-ground half of the table above."""
    rows = []
    for sp in (0.5, 0.9):
        r = mission(speed=sp)
        try:
            th = push_threshold(speed=sp)
        except RuntimeError:
            th = float("nan")
        rows.append((sp, r["path"], r["armerr"], th))
    return rows


if __name__ == "__main__":
    print("--- the capstone run: everything sections 1-9 established ---")
    r = mission()
    print(f"  walked {r['path']:.2f} m of path, x = {r['x']:.2f} m")
    print(f"  pelvis ended at {r['z']:.3f} m, fell = {r['fell']}")
    print(f"  arm held to {r['armerr']:.4f} rad, max tilt {r['tilt_max']:.4f}")
    print()
    print("  Every piece is one this project measured:")
    print("    6.x  the pre-trained policy, unmodified")
    print("    7.2  anti-phase arm swing, +0.25 m over bolted")
    print("    7.5  arm hold at kp=20, where push tolerance peaks")
    print("    8.2  command 0.9 m/s, because a slope is a force")
    print("    8.6  bisect for the push boundary, never one magnitude")
    print()

    print("--- and the two that conflict ---")
    print("  8.2: command MORE speed, a slope is a force, momentum helps.")
    print("  8.5: command LESS speed, a step is a timing error, momentum hurts.")
    print()
    print(f"  On flat ground, no conflict:")
    print(f"    {'speed':>7} {'path m':>8} {'arm err':>9} {'push N':>9}")
    for sp, pa, ae, th in conflict_table():
        print(f"    {sp:>7.1f} {pa:>8.2f} {ae:>9.4f} {th:>9.1f}")
    print("  0.9 wins on distance AND on push tolerance.")
    print()
    print("  On a 20 mm step run, 3 seeds each, it bites:")
    print(f"    {'speed':>7} {'survived':>10} {'mean x':>9}")
    print(f"    {0.3:>7.1f} {'3/3':>10} {1.75:>9.2f}")
    print(f"    {0.5:>7.1f} {'2/3':>10} {3.80:>9.2f}")
    print(f"    {0.9:>7.1f} {'2/3':>10} {10.12:>9.2f}")
    print()
    print("  Reliability and progress trade against each other, the exchange")
    print("  rate depends on terrain the robot cannot see, and there is no")
    print("  single speed that is right. That is the honest capstone result:")
    print("  not a system that works, but a system whose limits are known.")

#!/usr/bin/env python3
"""measure.py -- the measurement script the whole project leans on.

Loads the robot, holds a standing pose with the manufacturer's own PD gains,
and logs what ACTUALLY happens to a plain CSV with a header row.

Design rules, each one learned the hard way:
  * record what happened, not what you intended
  * runnable with no arguments and no setup
  * print enough that the output alone says whether the run was valid
  * REFUSE to produce a number when the assumptions are violated

Usage:  python3 measure.py [seconds] [out.csv]
"""
import csv, math, os, sys
import numpy as np
import mujoco

ROOT = os.path.expanduser("~/humanoid_ws")
SCENE = f"{ROOT}/mujoco/resources/robots/h1_2/scene.xml"

# The manufacturer's gains, from deploy/deploy_mujoco/configs/h1_2.yaml. Using
# their numbers rather than invented ones keeps our results comparable to theirs.
KP = np.array([200, 200, 200, 300, 40, 40, 200, 200, 200, 300, 40, 40], float)
KD = np.array([2.5, 2.5, 2.5, 4, 2, 2, 2.5, 2.5, 2.5, 4, 2, 2], float)
NOMINAL = np.array([0, -0.16, 0.0, 0.36, -0.2, 0.0,
                    0, -0.16, 0.0, 0.36, -0.2, 0.0], float)
DECIM = 10                       # log at 50 Hz against a 500 Hz physics step
FOOT_HALF = 0.12                 # half a 0.24 m foot: the CoP travel limit
FEET = ("left_ankle_roll_link", "right_ankle_roll_link")


def main(dur=8.0, out="stand.csv"):
    if not os.path.exists(SCENE):
        sys.exit("FATAL: scene not found: %s" % SCENE)
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    if m.nu != len(KP):
        sys.exit("FATAL: model has %d actuators, gains describe %d" % (m.nu, len(KP)))
    # Address each driven joint through the MODEL, never as 7 + actuator_index.
    # On this 12 DOF scene the two happen to agree; on the 51 actuator full body
    # they do not, and the mismatch reads as frozen joints rather than as a bug.
    QADR = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    VADR = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    foot = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in FEET]
    if any(b < 0 for b in foot):
        sys.exit("FATAL: foot bodies %s not in this model" % (FEET,))

    rows, step = [], 0
    n = int(dur / m.opt.timestep)
    while step < n:
        # the pose is HELD every step: this is position control doing its job
        d.ctrl[:] = KP * (NOMINAL - d.qpos[QADR]) - KD * d.qvel[VADR]
        mujoco.mj_step(m, d)
        step += 1
        if not np.all(np.isfinite(d.qpos)):
            sys.exit("FATAL: state went non-finite at t=%.3f; refusing to log" % d.time)
        if step % DECIM:
            continue
        com = d.subtree_com[0]
        # contact: are we actually ON the floor, or through it, or above it
        fz = 0.0
        for i in range(d.ncon):
            fr = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, fr)
            fz += fr[0]
        # How far the CoM has escaped the support region, horizontally.
        # Measure against the FEET, not against the pelvis. The pelvis is part
        # of the falling body, so a CoM-to-pelvis distance stays small however
        # far the robot pitches over, and it will report "never escaped" while
        # the torso is lying at sixty three degrees. Ask the ground, not the hips.
        mid = (d.xpos[foot[0]] + d.xpos[foot[1]]) / 2.0
        esc = math.hypot(com[0] - mid[0], com[1] - mid[1])
        rows.append(dict(t=round(d.time, 4), pelvis_z=round(float(d.qpos[2]), 4),
                         com_x=round(float(com[0]), 4), com_z=round(float(com[2]), 4),
                         ncon=int(d.ncon), fz=round(float(fz), 2),
                         com_offset=round(float(esc), 4),
                         track_err=round(float(np.abs(NOMINAL - d.qpos[QADR]).max()), 4)))

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    z0, zf = rows[0]["pelvis_z"], rows[-1]["pelvis_z"]
    zmin = min(r["pelvis_z"] for r in rows)
    esc_t = next((r["t"] for r in rows if r["com_offset"] > FOOT_HALF), None)
    worst_track = max(r["track_err"] for r in rows)
    print("wrote %s: %d samples over %.1f s" % (out, len(rows), dur))
    print("pelvis z: start %.3f  min %.3f  final %.3f" % (z0, zmin, zf))
    print("worst joint tracking error: %.4f rad (%.2f deg)"
          % (worst_track, math.degrees(worst_track)))
    print("CoM left the support region at t=%s s"
          % ("%.2f" % esc_t if esc_t else "never"))
    print("VERDICT: %s" % ("FELL" if zf < 0.6 else "STOOD"))


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 8.0,
         sys.argv[2] if len(sys.argv) > 2 else "stand.csv")

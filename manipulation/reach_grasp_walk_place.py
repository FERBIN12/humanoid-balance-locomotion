#!/usr/bin/env python3
"""7.7 -- the whole sequence: reach, grasp, walk, place. One continuous run.

Every previous experiment in this section did ONE thing. 7.3 grasped a stationary
object with the robot standing still. 7.4 carried a payload that was WELDED to
the wrist, which is to say it could not be dropped. 7.6 held an arm pose while
walking. This chains them, and the point of chaining them is that the handoffs
between phases are where it breaks, not the phases themselves.

Four phases, with the transition condition for each stated as a measurement
rather than a timer wherever one exists:

  1. REACH   solve the 7 DOF arm IK to a pre-grasp 45 mm short of the object
             (the offset is 7.3's, swept not chosen: solving to the object
             CENTRE puts the digits inside it and kicks it away at t=0)
  2. GRASP   close 24 finger joints, then remove the shelf. 7.3's window is
             350-800 g at 32 mm diameter, and it is a WINDOW: lighter objects
             are extruded upward like toothpaste because the closed hand still
             leaves a 16.92 mm gap and the digits never oppose.
  3. WALK    hand the legs to the the learned policy policy while the arm holds. The
             arm is no longer welded to anything, so the grasp has to survive
             the gait, which is the actual new question in this experiment.
  4. PLACE   stop, lower, open the fingers, and check the object is on the
             shelf and STAYS there.

Run it:  ~/humanoid_ws/rlvenv/bin/python reach_grasp_walk_place.py
"""
import os
import pathlib
import re

import mujoco
import numpy as np
import torch
import yaml

from reach_ik import (ARM, FINGERS, NAMES, SCENE, digit_centroid, ik_arm,
                      qadr, vadr)

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
PT = str(ROOT / "policy/pre_train/h1_2/motion.pt")
# NOT scene_grasp.xml. That file welds the pelvis to the world, because 7.3 is
# a hand experiment where a failed grasp must not be confused with a fallen
# robot. This experiment walks, and with that weld the policy moves the robot 8 mm
# in 8 s while the pelvis z never leaves 1.0300: a pinned base that reads
# exactly like a walking failure. scene_carry.xml is the same scene with the
# weld removed, so the robot can walk and can fall.
CARRY = str(ROOT / "mujoco/resources/robots/h1_2/scene_carry.xml")
BASE = pathlib.Path(CARRY).read_text()

KP = np.array(cfg["kps"], np.float32)
KD = np.array(cfg["kds"], np.float32)
DEFAULT = np.array(cfg["default_angles"], np.float32)
CMD = np.array(cfg["cmd_init"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]
NA = 12
GAIT = 0.8

# 7.3's measured graspable window: 350 to 800 g at 32 mm diameter.
RADIUS = 0.016          # 32 mm across
MASS = 0.500            # mid-window, so the grasp is not the marginal thing


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def ik_here(m, d, QA, VA, target, iters=300, lam=0.05, step=0.6):
    """Damped least squares for the 7 arm DOFs, solved IN THE CURRENT STATE.

    reach_ik.ik_arm cannot be reused here. It builds its own MjModel from
    SCENE and a fresh MjData, so it always solves against the pose the robot
    SPAWNS in, whatever state the caller is in. That is correct for 7.3, whose
    base is welded and never moves. Here the base drifts about 50 mm while the
    robot settles and steps in place, and a pose solved for the spawn base
    aimed the hand at where the object used to be relative to the robot: the
    IK reported 0.15 mm of error and the hand missed by 45 mm.

    So this solves on a COPY of the live state, leaving the caller's d alone.
    """
    from reach_ik import WRIST
    dw = mujoco.MjData(m)
    dw.qpos[:] = d.qpos
    dw.qvel[:] = 0.0
    jnts = [m.actuator_trnid[i][0] for i in ARM]
    lo = np.array([m.jnt_range[j][0] for j in jnts])
    hi = np.array([m.jnt_range[j][1] for j in jnts])
    jacp = np.zeros((3, m.nv)); jacr = np.zeros((3, m.nv))
    e = None
    for _ in range(iters):
        mujoco.mj_forward(m, dw)
        err = target - digit_centroid(m, dw)
        e = float(np.linalg.norm(err))
        if e < 2e-4:
            break
        mujoco.mj_jacBody(m, dw, jacp, jacr, WRIST)
        J = jacp[:, [VA[i] for i in ARM]]
        dq = J.T @ np.linalg.solve(J @ J.T + (lam ** 2) * np.eye(3), err)
        q = dw.qpos[[QA[i] for i in ARM]] + step * dq
        dw.qpos[[QA[i] for i in ARM]] = np.clip(q, lo, hi)
    mujoco.mj_forward(m, dw)
    return dict(q=dw.qpos[[QA[i] for i in ARM]].copy(), err=e,
                reached=digit_centroid(m, dw))


def variant(radius, tag):
    s = re.sub(r'size="0\.0\d+ 0\.0\d+"', 'size="%.4f 0.040"' % radius, BASE)
    p = ROOT / ("mujoco/resources/robots/h1_2/_v_%s.xml" % tag)
    p.write_text(s)
    m = mujoco.MjModel.from_xml_path(str(p))
    p.unlink()
    return m


def sequence(mass=MASS, radius=RADIUS, walk_for=6.0, close_s=1.2,
             verbose=True):
    """One continuous run through all four phases. Returns per-phase results.

    Nothing here is scripted from a timeline except the phase boundaries, and
    the object is a free body throughout: if the grasp fails the object falls,
    and the run records that instead of pretending otherwise.
    """
    CLOSE_S = close_s
    m = variant(radius, "s77")
    QA, VA = qadr(m), vadr(m)
    nonf = [i for i in range(m.nu) if i not in FINGERS]
    OBJ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    oq = m.jnt_qposadr[m.body_jntadr[OBJ]]
    OG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "object_geom")
    SG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "shelf_geom")
    FG = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    IGNORE = {SG, FG, OG}

    m.body_mass[OBJ] = mass
    d = mujoco.MjData(m)
    d.qpos[QA[:NA]] = DEFAULT
    mujoco.mj_forward(m, d)
    spawn_q = d.qpos[QA].copy()
    obj0 = d.qpos[oq:oq + 3].copy()
    z0 = float(d.qpos[oq + 2])
    # IK is solved LATER, against the settled base. Solving it here, at spawn,
    # produced a 0.15 mm accurate arm pose that still missed by 45 mm: the
    # object is fixed in WORLD coordinates and the robot's base drifts 50 mm
    # while it settles and steps in place, so a pose solved for the spawn base
    # points the hand at where the object used to be relative to the robot.
    hold = spawn_q.copy()
    solved = [False]

    policy = torch.jit.load(PT)
    policy.eval()
    action = np.zeros(NA, np.float32)
    obs = np.zeros(cfg["num_obs"], np.float32)
    target = DEFAULT.copy()

    # A SETTLE phase first. With the base free the robot drops 24 mm and drifts
    # 47 mm in the first half second, and driving the arm to the IK pose at 400
    # stiffness during that transient sweeps the hand through the object and
    # knocks it off the shelf before the fingers have closed. 7.3 never saw
    # this because its base was welded and nothing settled.
    T_SETTLE = 1.0
    T_GRASP = T_SETTLE + 2.0
    T_LIFT = T_GRASP + 1.6
    T_WALK = T_LIFT + walk_for
    T_END = T_WALK + 4.0
    dt = m.opt.timestep
    rec = {"contacts": [], "objz": [], "pelvz": [], "x": [], "phase": []}
    fell = None
    walk_x0 = None

    for k in range(int(T_END / dt)):
        t = k * dt
        if t < T_SETTLE:
            phase = "settle"
        elif t < T_GRASP:
            phase = "reach"
        elif t < T_LIFT:
            phase = "grasp"
        elif t < T_WALK:
            phase = "walk"
        else:
            phase = "place"

        # the shelf goes away once the fingers have closed, so "held" cannot
        # be satisfied by the object simply resting on something (7.3's bug)
        if t >= T_LIFT:
            m.geom_contype[SG] = 0
            m.geom_conaffinity[SG] = 0
        elif phase == "place" and t > T_WALK + 1.5:
            m.geom_contype[SG] = 1
            m.geom_conaffinity[SG] = 1

        # RE-SOLVE the IK continuously while reaching, at 20 Hz. A single solve
        # is not enough and this is the real lesson of the experiment: the object
        # is fixed in WORLD coordinates, and the policy keeps stepping in place
        # to stand, so the base drifts throughout. A one-shot solve reported
        # 0.19 mm of IK error and then let the hand slide from the intended
        # 45 mm pre-grasp out to 139 mm, because the pose was correct for a
        # base that had already moved on. Standing still is not standing still.
        if phase in ("reach", "grasp") and k % 100 == 0:
            objn = d.qpos[oq:oq + 3].copy()
            apn = digit_centroid(m, d) - objn
            apn /= np.linalg.norm(apn)
            sol = ik_here(m, d, QA, VA, objn + apn * 0.045, iters=60)
            hold = d.qpos[QA].copy()
            for j, ai in enumerate(ARM):
                hold[ai] = sol["q"][j]
            solved[0] = True

        q, dq = d.qpos[QA], d.qvel[VA]
        tau = np.zeros(m.nu)

        # The LEGS are always the policy's, in every phase. Holding them at a
        # fixed pose instead lets the whole robot topple slowly forward: the
        # pelvis drifted 132 mm in 2.5 s and the hand carried the object off a
        # shelf that is fixed in WORLD coordinates. Standing still is a control
        # problem, which is the balance controller's whole point, and a stiff joint hold is
        # not a stand.
        tau[:NA] = (target - q[:NA]) * KP - dq[:NA] * KD
        up = [i for i in nonf if i >= NA]
        if phase == "settle":
            tgt_q = spawn_q
        elif phase == "reach":
            # ease in over the first half second so the approach is a motion,
            # then track the freshly solved pose
            a = min(1.0, (t - T_SETTLE) / 0.5)
            tgt_q = spawn_q + (hold - spawn_q) * a
        else:
            tgt_q = hold
        tau[up] = (tgt_q[up] - q[up]) * 400.0 - dq[up] * 20.0

        # the fingers close from t=0.4 and stay closed until the place
        if phase == "place" and t > T_WALK + 2.0:
            g = max(0.0, 1.0 - (t - T_WALK - 2.0) / 1.0)     # open again
        else:
            # close only once the hand has ARRIVED. Closing during the reach
            # bats the object off the shelf.
            g = min(1.0, max(0.0, (t - T_GRASP) / CLOSE_S))
        for i in FINGERS:
            tg = g * (0.70 if "thumb_proximal_yaw" in NAMES[i] else 1.40)
            tau[i] = np.clip((tg - q[i]) * 3.0 - dq[i] * 0.1, -1, 1)

        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if k % DECIM == 0:
            ph = (t % GAIT) / GAIT
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_body(d.qpos[3:7])
            # zero velocity command in every phase but the walk, so the same
            # policy stands still and then walks without a controller swap
            obs[6:9] = (CMD if phase == "walk" else CMD * 0.0) * CMD_SCALE
            obs[9:9 + NA] = (d.qpos[QA[:NA]] - DEFAULT) * cfg["dof_pos_scale"]
            obs[9 + NA:9 + 2 * NA] = d.qvel[VA[:NA]] * cfg["dof_vel_scale"]
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)) \
                    .numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT

        if phase == "walk" and walk_x0 is None:
            walk_x0 = float(d.qpos[0])

        if k % 25 == 0:
            # EXCLUDE the floor and the shelf. An object lying on the ground
            # produces two steady contacts, and counting those reported a
            # "grasp" of 2.00 contacts for an object that was on the floor
            # through every phase.
            n = 0
            for ci in range(d.ncon):
                c = d.contact[ci]
                if OG in (c.geom1, c.geom2) and not {c.geom1, c.geom2} & IGNORE:
                    n += 1
            rec["contacts"].append(n)
            rec["objz"].append(float(d.qpos[oq + 2]))
            rec["pelvz"].append(float(d.qpos[2]))
            rec["x"].append(float(d.qpos[0]))
            rec["phase"].append(phase)

        if d.qpos[2] < 0.4 and fell is None:
            fell = t

    objz = np.array(rec["objz"])
    ph = np.array(rec["phase"])
    out = dict(mass=mass, fell=fell,
               z0=z0,
               walked=float(d.qpos[0] - (walk_x0 or 0.0)),
               obj_end=float(d.qpos[oq + 2]),
               drop=float(z0 - d.qpos[oq + 2]))
    for name in ("reach", "grasp", "walk", "place"):
        sel = ph == name
        if sel.any():
            c = np.array(rec["contacts"])[sel]
            out[name + "_contacts"] = float(c.mean())
            out[name + "_objz"] = float(objz[sel].mean())
    # held through the walk means contacts held AND the object did not fall
    out["carried"] = (out.get("walk_contacts", 0) > 0.5
                      and abs(out.get("walk_objz", 0) - z0) < 0.10
                      and fell is None)
    if verbose:
        print(f"  mass {mass * 1000:.0f} g   walked {out['walked']:.2f} m   "
              f"fell={out['fell']}")
        for name in ("reach", "grasp", "walk", "place"):
            if name + "_contacts" in out:
                print(f"    {name:>6}  contacts {out[name + '_contacts']:5.2f}"
                      f"   object z {out[name + '_objz']:.3f}")
        print(f"    carried through the walk: {out['carried']}")
    return out


def hand_drift(dur=10.0, seed=0):
    """How far does the hand move when the robot is STANDING STILL?

    The arm is held at a FIXED joint pose for the whole run and the velocity
    command is zero, so every millimetre the hand moves is base motion. This is
    the measurement the experiment turns on, and it is the one 7.3 could not make
    because its pelvis was welded to the world.
    """
    m = variant(RADIUS, "drift")
    QA, VA = qadr(m), vadr(m)
    d = mujoco.MjData(m)
    d.qpos[QA[:NA]] = DEFAULT
    mujoco.mj_forward(m, d)
    spawn = d.qpos[QA].copy()
    policy = torch.jit.load(PT); policy.eval()
    obs = np.zeros(cfg["num_obs"], np.float32)
    action = np.zeros(NA, np.float32); target = DEFAULT.copy()
    nonf = [i for i in range(m.nu) if i not in FINGERS]
    up = [i for i in nonf if i >= NA]
    dig, pel = [], []
    for k in range(int(dur / m.opt.timestep)):
        t = k * m.opt.timestep
        tau = np.zeros(m.nu)
        tau[:NA] = (target - d.qpos[QA[:NA]]) * KP - d.qvel[VA[:NA]] * KD
        tau[up] = (spawn[up] - d.qpos[QA][up]) * 400.0 - d.qvel[VA][up] * 20.0
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)
        if k % DECIM == 0:
            ph = (t % GAIT) / GAIT
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_body(d.qpos[3:7])
            obs[6:9] = 0.0                       # stand, do not walk
            obs[9:9 + NA] = (d.qpos[QA[:NA]] - DEFAULT) * cfg["dof_pos_scale"]
            obs[9 + NA:9 + 2 * NA] = d.qvel[VA[:NA]] * cfg["dof_vel_scale"]
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)).numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT
        if k % 25 == 0:
            dig.append(digit_centroid(m, d).copy())
            pel.append(d.qpos[:3].copy())
    dig = np.array(dig); pel = np.array(pel)
    step = np.linalg.norm(np.diff(dig, axis=0), axis=1)
    speed = float(np.median(step)) / 0.05
    return dict(pelvis_mm=1000 * float(np.linalg.norm(pel[-1, :2] - pel[0, :2])),
                hand_mm=1000 * float(np.linalg.norm(dig[-1] - dig[0])),
                speed_mm_s=1000 * speed,
                close_mm=1000 * speed * 1.2,
                widths=(1000 * speed * 1.2) / (2000 * RADIUS))


if __name__ == "__main__":
    print("--- first, the thing 7.3 could not measure ---")
    print("  7.3's scene WELDS the pelvis to the world, on purpose: it is a")
    print("  hand experiment, and a failed grasp must not be confused with a")
    print("  fallen robot. That weld is also why its grasp works. Remove it,")
    print("  hold the arm at a FIXED joint pose, command zero velocity, and")
    print("  ask how far the hand moves while the robot 'stands still'.")
    print()
    dr = hand_drift()
    print(f"  pelvis drift over 10 s        {dr['pelvis_mm']:.1f} mm")
    print(f"  hand drift over 10 s          {dr['hand_mm']:.1f} mm")
    print(f"  hand speed, median            {dr['speed_mm_s']:.1f} mm/s")
    print(f"  hand travel in a 1.2 s close  {dr['close_mm']:.1f} mm")
    print(f"  the object is {2000 * RADIUS:.0f} mm across, so the hand crosses")
    print(f"  {dr['widths']:.1f} OBJECT WIDTHS while the fingers are closing.")
    print()
    print("  Standing still is not standing still. The policy holds balance by")
    print("  stepping, which is the balance controller's lesson arriving somewhere new: the")
    print("  hand has no stationary frame to close in.")
    print()

    print("--- so does the sequence work? ---")
    r = sequence()
    print()

    print("--- is it just closing too slowly? ---")
    print("  If the failure is that the hand crosses 3 object widths during a")
    print("  1.2 s close, a faster close crosses fewer and should help.")
    print()
    print(f"  {'close s':>8} {'widths crossed':>15} {'object z at grasp':>18} "
          f"{'carried':>8}")
    for cs in (1.2, 0.5, 0.25):
        rr = sequence(close_s=cs, verbose=False)
        w = dr["speed_mm_s"] * cs / (2000 * RADIUS)
        print(f"  {cs:>8.2f} {w:>15.1f} {rr.get('grasp_objz', 0):>18.4f} "
              f"{str(rr['carried']):>8}")
    print()
    print("  It does not help, it hurts: the object is knocked further and")
    print("  sooner. So the failure is not the crossing DISTANCE. The hand is")
    print("  sweeping THROUGH the object rather than closing around it, which")
    print("  is 7.3's original finding restated: the digits never oppose, and")
    print("  a hand that cannot oppose needs its target to hold still. A")
    print("  welded base held it still. A balancing robot does not.")
    print()
    print("--- what this experiment actually establishes ---")
    print("  7.3's graspable window, 350 to 800 g at 32 mm, is a claim about a")
    print("  BOLTED robot. It does not transfer. Every phase here works on its")
    print("  own: the IK solves to 0.19 mm, the robot walks 2.5 m and stays")
    print("  up, the fingers close. The chain still fails, at the handoff, for")
    print("  a reason none of the phases could show you alone.")
    print("  That is what chaining is for, and it is why this is a experiment")
    print("  rather than a victory lap.")

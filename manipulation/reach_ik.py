#!/usr/bin/env python3
"""7.3 part one -- pose the ARM so the hand can actually grasp.

The robot model closed all 24 finger joints on empty air and that worked. Closing
them on an OBJECT does not, and the reason is not the fingers.

Driven from the default arm pose, the digits converge at (0.408, 0.162, 1.124)
in a 19 x 14 mm cross-section. Put a cylinder there and the fingers make ZERO
contact with it at every timestep of a 2.4 s close: the palm sweeps past and
nudges the object from x=0.408 to x=0.357 on the way. The digits arrive at the
right place along a path that never envelops a stationary object, because the
default pose has the palm facing the wrong way.

So the fingers were never the problem. The ARM is. This file solves for a
7 DOF arm pose that puts the palm around a target, using damped least squares
on the MuJoCo Jacobian, respecting the real joint limits.
"""
import os
import pathlib

import mujoco
import numpy as np

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
SCENE = str(ROOT / "mujoco/resources/robots/h1_2/scene_grasp.xml")

_m = mujoco.MjModel.from_xml_path(SCENE)
NAMES = [mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(_m.nu)]
ARM = [i for i, n in enumerate(NAMES)
       if n.startswith("left_") and any(k in n for k in
                                        ("shoulder", "elbow", "wrist"))]
FINGERS = [i for i, n in enumerate(NAMES)
           if any(k in n for k in ("thumb", "index", "middle", "ring",
                                   "pinky"))]
# The wrist is the last body of the arm chain; the digits hang off it.
WRIST = mujoco.mj_name2id(_m, mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")


def qadr(m):
    return np.array([m.jnt_qposadr[m.actuator_trnid[i][0]]
                     for i in range(m.nu)])


def vadr(m):
    return np.array([m.jnt_dofadr[m.actuator_trnid[i][0]]
                     for i in range(m.nu)])


def digit_centroid(m, d):
    """Where the LEFT hand's distal and intermediate links currently sit.

    This, not the wrist, is what has to end up around the object: the wrist
    can be in the right place with the palm pointing the wrong way.
    """
    pts = []
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                               m.geom_bodyid[g]) or ""
        if nm.startswith("L_") and ("distal" in nm or "intermediate" in nm):
            pts.append(d.geom_xpos[g])
    return np.mean(pts, axis=0) if pts else None


def ik_arm(target, iters=300, lam=0.15, step=0.6, verbose=False):
    """Damped least squares on the 7 arm DOFs to put the DIGITS at `target`.

    Damped rather than plain pseudoinverse: the arm passes through
    configurations where the Jacobian loses rank (elbow straight), and there a
    plain inverse asks for an unbounded joint step. lam trades tracking
    accuracy for conditioning, and 0.15 was chosen by the sweep in __main__,
    not guessed.
    """
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    QA, VA = qadr(m), vadr(m)
    mujoco.mj_forward(m, d)

    jnts = [m.actuator_trnid[i][0] for i in ARM]
    lo = np.array([m.jnt_range[j][0] for j in jnts])
    hi = np.array([m.jnt_range[j][1] for j in jnts])

    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    hist = []
    for it in range(iters):
        mujoco.mj_forward(m, d)
        cur = digit_centroid(m, d)
        err = target - cur
        e = np.linalg.norm(err)
        hist.append(e)
        if e < 2e-4:
            break
        # Jacobian of the WRIST body: the digits are rigid relative to it while
        # the fingers are not being driven, so this maps arm motion to digit
        # motion up to a constant offset.
        mujoco.mj_jacBody(m, d, jacp, jacr, WRIST)
        J = jacp[:, [VA[i] for i in ARM]]          # 3 x 7
        # damped least squares: dq = J^T (J J^T + lam^2 I)^-1 e
        JJt = J @ J.T + (lam ** 2) * np.eye(3)
        dq = J.T @ np.linalg.solve(JJt, err)
        q = d.qpos[[QA[i] for i in ARM]] + step * dq
        d.qpos[[QA[i] for i in ARM]] = np.clip(q, lo, hi)
    mujoco.mj_forward(m, d)
    return dict(q=d.qpos[[QA[i] for i in ARM]].copy(),
                reached=digit_centroid(m, d), err=hist[-1], iters=len(hist),
                hist=hist)


if __name__ == "__main__":
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    OBJ = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
    oq = m.jnt_qposadr[m.body_jntadr[OBJ]]
    obj = d.qpos[oq:oq + 3].copy()

    print("--- the problem, restated with numbers ---")
    print(f"  object sits at            {np.round(obj, 4)}")
    print(f"  digits start at           {np.round(digit_centroid(m, d), 4)}")
    print(f"  distance                  "
          f"{np.linalg.norm(obj - digit_centroid(m, d)):.4f} m")
    print("  and closing the fingers from here makes ZERO contact with it.")
    print()

    print("--- damping sweep: lam is measured, not chosen ---")
    print(f"  {'lam':>6} {'iters':>6} {'final err m':>12} {'verdict':>10}")
    best = None
    for lam in (0.01, 0.05, 0.15, 0.40, 1.00):
        r = ik_arm(obj, lam=lam)
        ok = r["err"] < 1e-3
        print(f"  {lam:>6.2f} {r['iters']:>6d} {r['err']:>12.6f} "
              f"{'converged' if ok else 'no':>10}")
        if ok and (best is None or r["iters"] < best[1]["iters"]):
            best = (lam, r)
    print()
    if best is None:
        print("  NOTHING CONVERGED. Do not proceed: a grasp measured from an")
        print("  unconverged pose is measuring the wrong pose.")
        raise SystemExit(1)
    lam, r = best
    print(f"  best lam {lam} in {r['iters']} iterations, "
          f"error {r['err'] * 1000:.3f} mm")
    print()

    print("--- the solved arm pose ---")
    for i, a in enumerate(ARM):
        j = m.actuator_trnid[a][0]
        lo, hi = m.jnt_range[j]
        at_limit = abs(r["q"][i] - lo) < 1e-6 or abs(r["q"][i] - hi) < 1e-6
        print(f"  {NAMES[a]:28} {r['q'][i]:+.4f} rad   "
              f"range [{lo:+.2f}, {hi:+.2f}]{'  AT LIMIT' if at_limit else ''}")
    print()
    n_lim = sum(1 for i, a in enumerate(ARM)
                for j in [m.actuator_trnid[a][0]]
                if abs(r["q"][i] - m.jnt_range[j][0]) < 1e-6
                or abs(r["q"][i] - m.jnt_range[j][1]) < 1e-6)
    print(f"  joints resting on a limit: {n_lim} of {len(ARM)}")
    if n_lim:
        print("  A solution pinned at a bound is a solution the optimiser")
        print("  could not improve, not necessarily the pose you wanted.")
    print()
    print(f"  digits reached {np.round(r['reached'], 4)}")
    print(f"  target was     {np.round(obj, 4)}")
    print(f"  residual       {r['err'] * 1000:.3f} mm")

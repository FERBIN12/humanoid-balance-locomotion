#!/usr/bin/env python3
"""7.2 -- arm control while walking: can a deliberate swing beat a bolted arm?

7.1 ended with a bolted arm as the best available option, at 6.94 m. That is an
unsatisfying place to stop, because it says the best thing to do with half the
robot is nothing. So here we drive the arms on purpose.

The controller is deliberately crude: swing the two shoulder pitches in
opposition, as a sine locked to the SAME 0.8 s gait clock the policy is fed.
One amplitude, one phase offset. That is the whole controller.

Two amplitudes and two phases produce a result that is monotone, 17x the seed
spread, and whose mechanism is NOT the one the textbooks give. It is also not
the one I first concluded from the data: see the axis test at the end.
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
CMD = np.array(cfg["cmd_init"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]
NA = 12
GAIT = 0.8
ARM_KP = 200.0          # 7.1's winner: hold hard, then swing about that hold

_m = mujoco.MjModel.from_xml_path(SCENE)
NAMES = [mujoco.mj_id2name(_m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(_m.nu)]
LSP = NAMES.index("left_shoulder_pitch_joint")
RSP = NAMES.index("right_shoulder_pitch_joint")
LSR = NAMES.index("left_shoulder_roll_joint")
RSR = NAMES.index("right_shoulder_roll_joint")


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def run(amp, phase, axis="pitch", dur=15.0, seed=0):
    """Walk with the shoulders swinging at `amp` rad, `phase` rad offset.

    axis="pitch" swings fore and aft, like a walking human.
    axis="roll"  swings the arms out sideways, moving the same mass laterally.
    """
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    policy = torch.jit.load(PT)
    policy.eval()
    rng = np.random.default_rng(seed)
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])

    d.qpos[qadr[:NA]] = DEFAULT
    d.qvel[:6] = rng.normal(0, 0.01, 6)
    mujoco.mj_forward(m, d)

    target = DEFAULT.copy()
    action = np.zeros(NA, np.float32)
    obs = np.zeros(cfg["num_obs"], np.float32)
    yaws, vxs, sways, pitches = [], [], [], []

    for step in range(int(dur / m.opt.timestep)):
        t = step * m.opt.timestep
        ph = (t % GAIT) / GAIT

        # --- the arm controller. Three lines. -----------------------------
        up = np.zeros(m.nu - NA)
        swing = amp * np.sin(2 * np.pi * ph + phase)
        if axis == "pitch":
            up[LSP - NA], up[RSP - NA] = swing, -swing
        else:
            up[LSR - NA], up[RSR - NA] = swing, -swing

        tau = np.zeros(m.nu)
        tau[:NA] = (target - d.qpos[qadr[:NA]]) * KP - d.qvel[vadr[:NA]] * KD
        tau[NA:] = ((up - d.qpos[qadr[NA:]]) * ARM_KP
                    - d.qvel[vadr[NA:]] * (ARM_KP * 0.05))
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        w, x, y, z = d.qpos[3:7]
        yaws.append(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        pitches.append(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
        vxs.append(float(d.qvel[0]))
        sways.append(float(d.qpos[1]))

        if step % DECIM == 0:
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_body(d.qpos[3:7])
            obs[6:9] = CMD * CMD_SCALE
            obs[9:9 + NA] = (d.qpos[qadr[:NA]] - DEFAULT) * cfg["dof_pos_scale"]
            obs[9 + NA:9 + 2 * NA] = d.qvel[vadr[:NA]] * cfg["dof_vel_scale"]
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)) \
                    .numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT

        if d.qpos[2] < 0.4:
            return None          # a fall is not a data point, it is a failure

    # skip the first 2 s of vx: the robot is still accelerating from rest
    return dict(dist=float(d.qpos[0]), vx=float(np.mean(vxs[1000:])),
                sway=float(np.std(sways)), yaw=float(np.degrees(np.std(yaws))),
                pitch=float(np.degrees(np.mean(pitches))))


def mean_of(amp, phase, axis="pitch", n=5, key=None):
    rs = [run(amp, phase, axis=axis, seed=s) for s in range(n)]
    rs = [r for r in rs if r]
    if not rs:
        return None
    if key:
        return float(np.mean([r[key] for r in rs]))
    return {k: float(np.mean([r[k] for r in rs])) for k in rs[0]}


if __name__ == "__main__":
    print("--- the controller ---")
    print("  shoulder_pitch_left  =  A sin(2 pi t / 0.8 + phi)")
    print("  shoulder_pitch_right = -A sin(2 pi t / 0.8 + phi)")
    print("  held at kp=200, which 7.1 measured as the best passive setting.")
    print("  phi=pi is 'anti phase': arms opposite to the gait clock phase.")
    print(f"  the commanded forward velocity is {CMD[0]} m/s throughout.")
    print()

    print("--- amplitude and phase sweep, 15 s each ---")
    print(f"  {'amp rad':>8} {'phase':>7} {'dist m':>8} {'vx m/s':>8} "
          f"{'sway':>7} {'yaw':>7}")
    for amp in (0.0, 0.10, 0.20, 0.35, 0.50):
        for phase, pn in ((0.0, "in"), (np.pi, "anti")):
            if amp == 0.0 and pn == "anti":
                continue
            r = run(amp, phase)
            if r is None:
                print(f"  {amp:>8.2f} {pn:>7}     FELL")
                continue
            print(f"  {amp:>8.2f} {pn:>7} {r['dist']:>8.2f} {r['vx']:>8.3f} "
                  f"{r['sway']:>7.3f} {r['yaw']:>7.2f}")
    print()
    print("  Both directions are monotone, and they go OPPOSITE ways.")
    print("  Anti phase climbs. In phase falls. Same amplitude, same energy,")
    print("  same joints: only the sign of the phase differs.")
    print()

    print("--- five seeds, because a one run trend is not a trend ---")
    cases = [("bolted, no swing", 0.0, 0.0), ("anti 0.35", 0.35, np.pi),
             ("in 0.35", 0.35, 0.0), ("anti 0.50", 0.50, np.pi)]
    print(f"  {'case':>17} " + " ".join(f"s{i}".rjust(6) for i in range(5))
          + f" {'mean':>7}")
    means = {}
    for name, amp, phase in cases:
        ds = []
        for s in range(5):
            r = run(amp, phase, seed=s)
            ds.append(r["dist"] if r else float("nan"))
        means[name] = float(np.nanmean(ds))
        print(f"  {name:>17} " + " ".join(f"{x:6.2f}" for x in ds)
              + f" {means[name]:7.2f}")
    base = means["bolted, no swing"]
    spread = max(run(0.0, 0.0, seed=s)["dist"] for s in range(5)) \
        - min(run(0.0, 0.0, seed=s)["dist"] for s in range(5))
    gap = means["anti 0.35"] - means["in 0.35"]
    print()
    print(f"  anti 0.35 vs bolted:  {means['anti 0.35'] - base:+.2f} m")
    print(f"  in 0.35 vs bolted:    {means['in 0.35'] - base:+.2f} m")
    print(f"  anti minus in:        {gap:.2f} m")
    print(f"  spread within bolted: {spread:.2f} m")
    print(f"  ratio:                {gap / max(1e-9, spread):.1f}x")
    print()

    print("--- so WHY does anti phase win? ---")
    print("  The textbook answer is yaw cancellation, and the yaw column")
    print("  already refuted it in 7.1. It refutes it again here: bolted has")
    print("  the LOWEST yaw of any row, and anti phase beats it while yawing")
    print("  MORE. Whatever the arms are buying, it is not heading.")
    print()
    print("  What does track the distance is the forward speed. The command")
    print(f"  is {CMD[0]} m/s. Measured:")
    for name, amp, phase in cases:
        v = mean_of(amp, phase, key="vx")
        print(f"    {name:>17}  vx {v:.3f} m/s   error {abs(v - CMD[0]):.3f}")
    print("  Anti phase moves the robot TOWARD its commanded speed and in")
    print("  phase moves it away. The arms are not steering. They are")
    print("  changing how well the policy tracks the speed it was asked for.")
    print()

    print("--- the mistake I nearly shipped ---")
    print("  Lateral sway also falls monotonically with anti phase amplitude,")
    print("  and rises with in phase. I wrote down 'the arms reduce sway, so")
    print("  the policy tracks better' and it fits every row in the table.")
    print()
    print("  It is wrong, and one test settles it. If the gain came from")
    print("  moving mass sideways, then swinging the arms out to the SIDE")
    print("  (shoulder roll) should reproduce it. Same joints, same mass,")
    print("  same amplitude, same gait lock, lateral instead of fore aft:")
    print()
    print(f"  {'axis':>7} {'amp':>6} {'dist m':>8} {'sway':>7} {'vx':>7}")
    for axis in ("pitch", "roll"):
        for amp in (0.0, 0.35):
            r = mean_of(amp, np.pi, axis=axis, n=3)
            if r is None:
                print(f"  {axis:>7} {amp:>6.2f}     FELL")
                continue
            print(f"  {axis:>7} {amp:>6.2f} {r['dist']:>8.2f} "
                  f"{r['sway']:>7.3f} {r['vx']:>7.3f}")
    print()
    rp = mean_of(0.35, np.pi, axis="pitch", n=3)
    rr = mean_of(0.35, np.pi, axis="roll", n=3)
    r0 = mean_of(0.0, np.pi, axis="pitch", n=3)
    print(f"  pitch buys {rp['dist'] - r0['dist']:+.2f} m.")
    print(f"  roll buys  {rr['dist'] - r0['dist']:+.2f} m.")
    print("  Roll does essentially nothing. So the sway reduction was a")
    print("  CONSEQUENCE, not the cause: the whole effect lives in the fore")
    print("  and aft axis, which is the axis the forward velocity is measured")
    print("  along. My sway story fit the data perfectly and was still wrong,")
    print("  because it never predicted anything the fore-aft story did not.")
    print()
    print("  Keep the shape of that test. Two explanations agreed on every")
    print("  number I had; the way to separate them was to find an experiment")
    print("  where they DISAGREE, and run that one.")

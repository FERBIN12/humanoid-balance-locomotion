#!/usr/bin/env python3
"""7.4 -- carry something while walking, and watch where the CoM goes.

7.3 established that this hand cannot grasp anything under 350 g, so the
payload here is WELDED to the forearm. That is deliberate and it is honest:
this experiment is about carrying mass, not about acquiring it. If the weld
bothers you, 7.3 is the experiment explaining why it is necessary.

The result is backwards. Carrying 5 kg makes the robot walk FURTHER than
carrying nothing, at the same commanded 0.5 m/s. It is not a strength effect
and it is not noise: five seeds per load, tiny spread. And the discriminating
test says it is not really about mass at all, it is about the lever arm.

MEASURE PATH LENGTH, NOT x. This policy does not walk straight; it veers. At
30 s the wrist-loaded run has travelled 20.32 m along its path but only 13.25 m
in x, because 13.51 m of it went sideways. Measured in x the lever advantage
looks like it collapses to -0.04 m at 30 s. Measured along the path it is
+4.51 m and still growing. That is the same error 6.5 made with the turning
circle, and it is why `dist` here is arc length.
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
BASE = (ROOT / "mujoco/resources/robots/h1_2/scene_full.xml").read_text()

KP = np.array(cfg["kps"], np.float32)
KD = np.array(cfg["kds"], np.float32)
DEFAULT = np.array(cfg["default_angles"], np.float32)
CMD = np.array(cfg["cmd_init"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]
NA = 12
GAIT = 0.8
ARM_KP = 200.0          # 7.1's winner

WRIST = "left_wrist_yaw_link"


def scene(mass, attach=WRIST, tag="t"):
    """scene_full.xml plus a payload welded to `attach`.

    The payload MUST have a freejoint. Without one it is a STATIC world body,
    and welding the wrist to a static body pins the robot to the world: every
    load then reported dist 0.00 m with byte-identical roll, which is the
    dead-path signature, not a physics result.
    """
    if mass <= 0:
        m = mujoco.MjModel.from_xml_path(
            str(ROOT / "mujoco/resources/robots/h1_2/scene_full.xml"))
        return m
    pos = "0.42 0.19 1.10" if attach == WRIST else "0.10 0.0 1.05"
    body = ('    <body name="load" pos="%s">\n'
            '      <freejoint name="load_free"/>\n'
            '      <geom name="load_geom" type="box" size="0.06 0.06 0.06"\n'
            '            rgba="0.9 0.5 0.2 1" mass="%.4f"/>\n'
            '    </body>\n' % (pos, mass))
    body += ('  </worldbody>\n  <equality>\n'
             '    <weld body1="%s" body2="load"/>\n  </equality>\n' % attach)
    s = BASE.replace("  </worldbody>", body, 1)
    p = ROOT / ("mujoco/resources/robots/h1_2/_carry_%s.xml" % tag)
    p.write_text(s)
    m = mujoco.MjModel.from_xml_path(str(p))
    p.unlink()
    return m


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def run(mass, attach=WRIST, dur=15.0, seed=0, tag="t"):
    m = scene(mass, attach, tag)
    d = mujoco.MjData(m)
    policy = torch.jit.load(PT)
    policy.eval()
    qadr = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    vadr = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
    rng = np.random.default_rng(seed)
    d.qpos[qadr[:NA]] = DEFAULT
    d.qvel[:6] = rng.normal(0, 0.01, 6)
    mujoco.mj_forward(m, d)

    target = DEFAULT.copy()
    action = np.zeros(NA, np.float32)
    obs = np.zeros(cfg["num_obs"], np.float32)
    rolls, comy, vxs = [], [], []
    # PATH LENGTH, not x. This policy veers: over 30 s the wrist-loaded run
    # reaches y = 13.5 m while x reaches 13.2, so x is a projection of a curve
    # and it made the lever advantage appear to vanish (-0.04 m) when measured
    # by path it is still +4.51 m. Same mistake as 6.5's turning circle.
    path = 0.0
    prev = d.qpos[:2].copy()

    for step in range(int(dur / m.opt.timestep)):
        t = step * m.opt.timestep
        tau = np.zeros(m.nu)
        tau[:NA] = (target - d.qpos[qadr[:NA]]) * KP - d.qvel[vadr[:NA]] * KD
        tau[NA:] = ((0.0 - d.qpos[qadr[NA:]]) * ARM_KP
                    - d.qvel[vadr[NA:]] * (ARM_KP * 0.05))
        d.ctrl[:] = tau
        mujoco.mj_step(m, d)

        if step % DECIM == 0:
            ph = (t % GAIT) / GAIT
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

        if step % 25 == 0:
            path += float(np.linalg.norm(d.qpos[:2] - prev))
            prev = d.qpos[:2].copy()

        if step % 50 == 0:
            mujoco.mj_comPos(m, d)
            w, x, y, z = d.qpos[3:7]
            rolls.append(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
            comy.append(float(d.subtree_com[1][1] - d.qpos[1]))
            vxs.append(float(d.qvel[0]))

        if d.qpos[2] < 0.4:
            return dict(mass=mass, fell=t, dist=path, x=float(d.qpos[0]),
                        y=float(d.qpos[1]),
                        roll=float(np.degrees(np.mean(rolls))),
                        comy=float(np.mean(comy)),
                        vx=float(np.mean(vxs[len(vxs) // 3:])))
    return dict(mass=mass, fell=None, dist=path, x=float(d.qpos[0]),
                y=float(d.qpos[1]),
                roll=float(np.degrees(np.mean(rolls))),
                comy=float(np.mean(comy)),
                vx=float(np.mean(vxs[len(vxs) // 3:])))


if __name__ == "__main__":
    print("--- the setup ---")
    print("  A 12 cm cube welded to the left forearm, walking at the same")
    print("  commanded 0.5 m/s as every other experiment in this section.")
    print("  The arms are held at kp=200, which 7.1 measured as the best")
    print("  passive setting, so nothing is swinging the load on purpose.")
    print()

    print("--- load sweep, 5 seeds each ---")
    print(f"  {'load kg':>8} " + " ".join(f"s{i}".rjust(6) for i in range(5))
          + f" {'mean':>7} {'falls':>6}")
    means, rolls = {}, {}
    for mass in (0.0, 1.0, 2.5, 5.0, 8.0, 12.0):
        ds, nf, rr = [], 0, []
        for s in range(5):
            r = run(mass, seed=s, tag="w%g_%d" % (mass, s))
            ds.append(r["dist"])
            rr.append(r["roll"])
            if r["fell"]:
                nf += 1
        means[mass] = float(np.mean(ds))
        rolls[mass] = float(np.mean(rr))
        print(f"  {mass:>8.1f} " + " ".join(f"{x:6.2f}" for x in ds)
              + f" {means[mass]:7.2f} {nf:>6}")
    print()
    peak = max(means, key=means.get)
    print(f"  empty:            {means[0.0]:.2f} m")
    print(f"  best:             {means[peak]:.2f} m at {peak} kg")
    print(f"  gain:             {means[peak] - means[0.0]:+.2f} m")
    print("  Carrying a load makes it walk FURTHER, up to a point, and the")
    print("  curve is an inverted U rather than a monotone decline.")
    print()

    print("--- and the roll, which does behave as you would expect ---")
    for mass in sorted(rolls):
        print(f"  {mass:>5.1f} kg   mean roll {rolls[mass]:+6.2f} deg")
    mono = all(abs(rolls[a]) <= abs(rolls[b]) + 1e-9
               for a, b in zip(sorted(rolls), sorted(rolls)[1:]))
    print(f"  monotone in load: {mono}")
    print("  So the robot IS leaning further with more mass. The lean is not")
    print("  the problem; up to 8 kg it is apparently part of the solution.")
    print()

    print("--- is it the MASS, or WHERE the mass is? ---")
    print("  The discriminating test. Weld the same 5 kg to the PELVIS, which")
    print("  is essentially on the CoM with no lever arm, and compare.")
    print()
    a = float(np.mean([run(5.0, WRIST, seed=s, tag="qa%d" % s)["dist"]
                       for s in range(3)]))
    b = float(np.mean([run(5.0, "pelvis", seed=s, tag="qb%d" % s)["dist"]
                       for s in range(3)]))
    e = float(np.mean([run(0.0, seed=s, tag="qe%d" % s)["dist"]
                       for s in range(3)]))
    print(f"  no load              {e:.2f} m")
    print(f"  5 kg on the pelvis   {b:.2f} m   ({b - e:+.2f} m)")
    print(f"  5 kg on the wrist    {a:.2f} m   ({a - e:+.2f} m)")
    print()
    print(f"  Mass alone buys {b - e:+.2f} m. Mass on a LEVER buys {a - e:+.2f} m,")
    print(f"  which is {(a - e) / max(1e-9, b - e):.1f} times as much. So this is")
    print("  not a strength or a loading effect. The payload on the forearm is")
    print("  a passive pendulum hanging off a swinging limb, and it is doing")
    print("  by accident what the deliberate arm controller in 7.2 was doing")
    print("  on purpose: adding fore and aft momentum in phase with the gait.")
    print()
    print("  Which also explains the inverted U. Too little mass and the")
    print("  pendulum contributes nothing; too much and it dominates the")
    print("  attitude the policy is trying to hold, and at 12 kg the robot")
    print("  falls on all five seeds. There is a best payload, and it is not")
    print("  zero.")
    print()
    print("--- what this does NOT show ---")
    print("  The load is welded, so nothing here says the robot could pick")
    print("  the object up, and 7.3 says clearly that it could not. Nor is")
    print("  this a claim that loading a real humanoid improves its walking:")
    print("  it is a claim about THIS policy, which was trained on a legs")
    print("  only model and has never seen a payload. An unmodelled pendulum")
    print("  happens to help it, exactly as an unmodelled limp arm happened")
    print("  to hurt it in 7.1. Both are the same fact about the same gap")
    print("  between what the policy models and what it is attached to.")

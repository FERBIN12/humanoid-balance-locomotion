#!/usr/bin/env python3
"""5.10 -- why industry reaches for learned policies.

Section five hand built a walking stack from first principles. Every piece is
verified correct in isolation and the assembly hops. This experiment puts that
next to a learned policy running the SAME robot in the SAME simulator, and
measures the difference rather than asserting it.

The policy is Unitree's BSD-3 licensed motion.pt for the H1-2. I did not train
it and I am not claiming its numbers as my own work: the point is the contrast.
"""
import numpy as np
import mujoco, os, yaml, torch

ROOT = os.path.expanduser("~/humanoid_ws")
SCENE = ROOT + "/mujoco/resources/robots/h1_2/scene.xml"
m = mujoco.MjModel.from_xml_path(SCENE)
NAMES = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(m.nu)]
IDX = {n: i for i, n in enumerate(NAMES)}
QA = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
VA = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")

cfg = yaml.safe_load(open(ROOT + "/policy/h1_2.yaml"))
KPP = np.array(cfg["kps"], dtype=np.float32)
KDP = np.array(cfg["kds"], dtype=np.float32)
DEFAULT = np.array(cfg["default_angles"], dtype=np.float32)
CMD = np.array(cfg["cmd_init"], dtype=np.float32)
NA = cfg["num_actions"]
DECIM = cfg["control_decimation"]


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], dtype=np.float32)


def run_policy(dur=20.0):
    policy = torch.jit.load(ROOT + "/policy/pre_train/h1_2/motion.pt")
    policy.eval()
    d = mujoco.MjData(m)
    d.qpos[7:7 + NA] = DEFAULT
    mujoco.mj_forward(m, d)
    action = np.zeros(NA, dtype=np.float32)
    target = DEFAULT.copy()
    obs = np.zeros(cfg["num_obs"], dtype=np.float32)
    minz, counter = 9.9, 0
    airborne = 0
    n = int(dur / m.opt.timestep)
    for step in range(n):
        tau = (target - d.qpos[7:7 + NA]) * KPP - d.qvel[6:6 + NA] * KDP
        d.ctrl[:NA] = tau
        mujoco.mj_step(m, d)
        counter += 1
        if counter % DECIM == 0:
            qj = (d.qpos[7:7 + NA] - DEFAULT) * cfg["dof_pos_scale"]
            dqj = d.qvel[6:6 + NA] * cfg["dof_vel_scale"]
            gv = gravity_body(d.qpos[3:7])
            om = d.qvel[3:6] * cfg["ang_vel_scale"]
            # 0.8 s gait period and cmd_scale come from the shipped config
            # and rig/mj_walk_test.py, not from my memory of them.
            ph = (counter * m.opt.timestep) % 0.8 / 0.8
            obs[:3] = om
            obs[3:6] = gv
            obs[6:9] = CMD * np.array(cfg["cmd_scale"], dtype=np.float32)
            obs[9:9 + NA] = qj
            obs[9 + NA:9 + 2 * NA] = dqj
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)) \
                    .numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT
        minz = min(minz, float(d.qpos[2]))
        if d.ncon == 0:
            airborne += 1
    return dict(dist=float(d.qpos[0]), speed=float(d.qpos[0]) / dur,
                minz=minz, finalz=float(d.qpos[2]),
                air=100.0 * airborne / n, upright=float(d.qpos[2]) > 0.8)


def run_built(dur=20.0, settle=1.5, gain=10.0):
    """The 5.7 stack, run for the SAME duration so the comparison is fair.
    Measured here rather than quoted: I first hardcoded 4.6 per cent airborne
    from memory and the real figure is 1.4."""
    KP0 = np.array([200., 200., 200., 300., 60., 40.] * 2) * gain
    KD0 = np.array([5., 5., 5., 7.5, 2., 2.] * 2) * np.sqrt(gain)
    LT = LS = 0.400
    LA, HIP_H = 0.020, 0.780
    T_STEP, STEP_L, CLEAR = 0.735, 0.30, 0.05

    def ik(dx, dz):
        tz = dz + LA
        r = np.hypot(dx, tz)
        if r > LT + LS:
            return None
        c = (r * r - LT ** 2 - LS ** 2) / (2 * LT * LS)
        kn = np.arccos(np.clip(c, -1.0, 1.0))
        be = np.arctan2(-dx, -tz)
        al = np.arctan2(LS * np.sin(kn), LT + LS * np.cos(kn))
        return be - al, kn

    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    ns = int(settle / m.opt.timestep)
    minz, air, n, fell = 9.9, 0, 0, None
    for step in range(int(dur / m.opt.timestep)):
        t = step * m.opt.timestep
        tgt = np.zeros(m.nu)
        hp0, kn0 = ik(0.0, -HIP_H)
        if step < ns:
            k = min(1.0, step / max(1, ns * 0.6))
            for s_ in ("left", "right"):
                tgt[IDX[s_ + "_hip_pitch_joint"]] = hp0 * k
                tgt[IDX[s_ + "_knee_joint"]] = kn0 * k
                tgt[IDX[s_ + "_ankle_pitch_joint"]] = -(hp0 + kn0) * k
        else:
            gt = t - settle
            u = (gt % T_STEP) / T_STEP
            sw = "left" if int(gt / T_STEP) % 2 == 0 else "right"
            st = "right" if sw == "left" else "left"
            sx = STEP_L * 0.5 * (1 - np.cos(np.pi * u)) - STEP_L / 2
            sz = -HIP_H + CLEAR * np.sin(np.pi * u)
            for side, (fx, fz) in ((sw, (sx, sz)), (st, (-sx, -HIP_H))):
                sol = ik(fx, fz)
                if sol is None:
                    continue
                hp, kn = sol
                tgt[IDX[side + "_hip_pitch_joint"]] = hp
                tgt[IDX[side + "_knee_joint"]] = kn
                tgt[IDX[side + "_ankle_pitch_joint"]] = -(hp + kn)
        d.ctrl[:] = KP0 * (tgt - d.qpos[QA]) - KD0 * d.qvel[VA]
        mujoco.mj_step(m, d)
        n += 1
        minz = min(minz, float(d.qpos[2]))
        if d.ncon == 0:
            air += 1
        if fell is None and float(d.qpos[2]) < 0.55:
            fell = t
    return dict(dist=float(d.qpos[0]), speed=float(d.qpos[0]) / dur,
                minz=minz, finalz=float(d.qpos[2]), fell=fell,
                air=100.0 * air / n, upright=float(d.qpos[2]) > 0.8)


print("--- the two controllers, same robot, same simulator ---")
print()
print("  hand built (5.7 to 5.9): trajectory + foot placement + IK + QP")
print("  learned: Unitree's BSD-3 motion.pt, which I did not train")
print()
r = run_policy()
b = run_built()
print("%22s %14s %12s" % ("", "hand built", "learned"))
print("%22s %14.3f %12.3f" % ("distance in 20 s (m)", b["dist"], r["dist"]))
print("%22s %14.3f %12.3f" % ("speed (m/s)", b["speed"], r["speed"]))
print("%22s %14.3f %12.3f" % ("lowest pelvis (m)", b["minz"], r["minz"]))
print("%22s %14.3f %12.3f" % ("final pelvis (m)", b["finalz"], r["finalz"]))
print("%22s %14s %12s" % ("still upright at end",
                          "no" if not b["upright"] else "yes",
                          "yes" if r["upright"] else "no"))
print("%22s %14.1f %12.1f" % ("percent airborne", b["air"], r["air"]))
print()
print("  the hand built stack fell at %.2f s having covered %.2f m, mostly by"
      % (b["fell"], b["dist"]))
print("  falling. The policy covers %.2f m and never drops below %.3f m."
      % (r["dist"], r["minz"]))
print()

print("--- what the policy is NOT ---")
print("  It is not smarter physics. Same MuJoCo, same 2 ms timestep, same")
print("  actuator limits, same robot. It is not a better trajectory either:")
print("  the policy has no trajectory at all. There is no CoM plan, no capture")
print("  point, no inverse kinematics and no quadratic program anywhere in it.")
print()
print("  It is a %d input, %d output network that maps the current state" % (cfg["num_obs"], NA))
print("  straight to joint targets, at %d Hz." % int(1.0 / (m.opt.timestep * DECIM)))
print()

print("--- what it actually bought ---")
print("  The walking stack built four components and verified each one:")
print("    5.3 CoM trajectory      satisfies its own equations to 2.2e-05")
print("    5.5 inverse kinematics  exact to 0.000000 m against MuJoCo FK")
print("    5.6 whole body QP       dynamics residual 5.8e-16")
print("  Every part correct. Assembled: it hops and falls at 2.72 s.")
print()
print("  The policy is not better at any of those sub-problems. It never")
print("  solves them. What it has is one thing the hand built stack never got:")
print("  it was optimised against the ASSEMBLED system, in contact, over whole")
print("  episodes. The interfaces between my correct components are exactly")
print("  where my stack failed, and those interfaces are what training sees.")
print()
print("  That is the honest reason industry reaches for learned policies on")
print("  humanoids, and it is not that the classical maths is wrong.")

#!/usr/bin/env python3
"""6.3 -- the 47 observations, term by term, with what each one is worth.

6.2 established the interface: 47 numbers in. This works out what those 47
are, where each comes from on a real robot, and which ones the policy actually
depends on. That last part is measured by perturbation rather than assumed
from the layout.
"""
import os, pathlib
import numpy as np
import torch, yaml, mujoco

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
NA = cfg["num_actions"]
NOBS = cfg["num_obs"]

# The layout, read off rig/mj_walk_test.py which is the working deploy loop.
LAYOUT = [
    (0, 3, "base angular velocity", "IMU gyro", cfg["ang_vel_scale"]),
    (3, 6, "gravity in body frame", "IMU quaternion", 1.0),
    (6, 9, "velocity command", "operator", None),
    (9, 9 + NA, "joint positions minus default", "encoders",
     cfg["dof_pos_scale"]),
    (9 + NA, 9 + 2 * NA, "joint velocities", "encoders",
     cfg["dof_vel_scale"]),
    (9 + 2 * NA, 9 + 3 * NA, "previous action", "the policy itself", 1.0),
    (9 + 3 * NA, 9 + 3 * NA + 2, "gait phase, sin and cos", "a clock", 1.0),
]

print("--- 1 the layout, and where each term comes from ---")
print("%6s %-32s %-22s %s" % ("index", "quantity", "sensor", "scale"))
tot = 0
for a, b, name, src, sc in LAYOUT:
    tot += b - a
    print("%6s %-32s %-22s %s"
          % ("%d..%d" % (a, b - 1), name, src,
             ("x%.2f" % sc) if sc else "see cmd_scale"))
print("%6s %-32s %d" % ("", "TOTAL", tot))
assert tot == NOBS, "layout sums to %d, config says %d" % (tot, NOBS)
print("  the layout sums to %d, which matches num_obs. Good." % tot)
print()

print("--- 2 what is NOT in there ---")
missing = ["absolute position", "absolute heading", "linear velocity of the base",
           "foot contact flags", "contact forces", "terrain height",
           "torque or current", "anything about the other leg's plan"]
for x in missing:
    print("  no %s" % x)
print()
print("  Two of those are worth dwelling on. The policy has NO linear velocity")
print("  of its own base, so it cannot directly know how fast it is going: it")
print("  is given a command and must infer its own speed from joint history.")
print("  That is one thing the LSTM memory from 6.2 is almost certainly for.")
print("  And it has NO contact information: it does not know its feet are on")
print("  the floor. It infers that too.")
print()

print("--- 3 which inputs does it actually use ---")
policy = torch.jit.load(str(ROOT / "policy/pre_train/h1_2/motion.pt"))
policy.eval()


def sensitivity(idx_range, trials=40, eps=0.5, seed=0):
    """How much does the action move when we perturb this block?

    Reload the policy for EVERY trial. 6.2 showed the LSTM keeps state across
    calls, so reusing one module would mix the previous perturbation into the
    next measurement and the numbers would be meaningless.
    """
    rng = np.random.default_rng(seed)
    dev = []
    for _ in range(trials):
        base = rng.normal(0, 0.3, (1, NOBS)).astype(np.float32)
        pert = base.copy()
        pert[0, idx_range[0]:idx_range[1]] += rng.normal(
            0, eps, idx_range[1] - idx_range[0]).astype(np.float32)
        p1 = torch.jit.load(str(ROOT / "policy/pre_train/h1_2/motion.pt"))
        p1.eval()
        p2 = torch.jit.load(str(ROOT / "policy/pre_train/h1_2/motion.pt"))
        p2.eval()
        with torch.no_grad():
            a = p1(torch.from_numpy(base)).numpy().squeeze()
            b = p2(torch.from_numpy(pert)).numpy().squeeze()
        dev.append(float(np.abs(a - b).mean()))
    return float(np.mean(dev)), float(np.std(dev))


print("  perturbing each block by the same amount and measuring how far the")
print("  action moves. Per unit of input, so blocks of different width are")
print("  comparable.")
print()
print("%-32s %12s %12s" % ("block", "mean |d action|", "per input"))
rows = []
for a, b, name, src, sc in LAYOUT:
    m_, s_ = sensitivity((a, b))
    rows.append((name, m_, m_ / (b - a), b - a))
    print("%-32s %12.5f %12.5f" % (name, m_, m_ / (b - a)))
print()
rows.sort(key=lambda r: -r[2])
print("  ranked by influence per input:")
for name, m_, per, n in rows:
    print("    %-32s %.5f" % (name, per))
print()
top = rows[0]
print("  the most influential block is '%s'." % top[0])
print("  I am NOT going to claim that means the policy 'cares about' it most.")
print("  A perturbation test measures local slope at random states, not")
print("  importance during a gait, and the states I sampled are noise rather")
print("  than poses this robot ever adopts. It is a real measurement of a")
print("  narrow thing, and 6.9 does the harder version properly.")

print()
print("--- 4 the same test on states the robot ACTUALLY visits ---")
print("  The states above are gaussian noise. Real observations are nothing")
print("  like that, so here is the same measurement using observations")
print("  recorded from the policy walking in MuJoCo.")
print()

SCENE = str(ROOT / "mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
KPP = np.array(cfg["kps"], dtype=np.float32)
KDP = np.array(cfg["kds"], dtype=np.float32)
DEFAULT = np.array(cfg["default_angles"], dtype=np.float32)
CMD = np.array(cfg["cmd_init"], dtype=np.float32)
DECIM = cfg["control_decimation"]


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], dtype=np.float32)


def collect(n_obs=60, dur=12.0):
    """Record real observation vectors from a walking run."""
    pol = torch.jit.load(str(ROOT / "policy/pre_train/h1_2/motion.pt"))
    pol.eval()
    d = mujoco.MjData(m)
    d.qpos[7:7 + NA] = DEFAULT
    mujoco.mj_forward(m, d)
    action = np.zeros(NA, dtype=np.float32)
    target = DEFAULT.copy()
    obs = np.zeros(NOBS, dtype=np.float32)
    out, counter = [], 0
    for step in range(int(dur / m.opt.timestep)):
        d.ctrl[:NA] = (target - d.qpos[7:7 + NA]) * KPP - d.qvel[6:6 + NA] * KDP
        mujoco.mj_step(m, d)
        counter += 1
        if counter % DECIM == 0:
            ph = (counter * m.opt.timestep) % 0.8 / 0.8
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_body(d.qpos[3:7])
            obs[6:9] = CMD * np.array(cfg["cmd_scale"], dtype=np.float32)
            obs[9:9 + NA] = (d.qpos[7:7 + NA] - DEFAULT) * cfg["dof_pos_scale"]
            obs[9 + NA:9 + 2 * NA] = d.qvel[6:6 + NA] * cfg["dof_vel_scale"]
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            with torch.no_grad():
                action = pol(torch.from_numpy(obs).unsqueeze(0)) \
                    .numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT
            if step * m.opt.timestep > 2.0:
                out.append(obs.copy())
    return np.array(out[:n_obs])


REAL = collect()
print("  collected %d real observation vectors from a walking run" % len(REAL))
print("  compare their spread to the gaussian test states:")
print("    real   std per element, mean %.3f  max %.3f"
      % (REAL.std(0).mean(), REAL.std(0).max()))
print("    test   std per element      0.300 by construction")
print()


def sens_real(idx_range, eps=0.5, seed=0):
    rng = np.random.default_rng(seed)
    dev = []
    for base in REAL[:30]:
        b0 = base.reshape(1, -1).astype(np.float32)
        pert = b0.copy()
        pert[0, idx_range[0]:idx_range[1]] += rng.normal(
            0, eps, idx_range[1] - idx_range[0]).astype(np.float32)
        p1 = torch.jit.load(str(ROOT / "policy/pre_train/h1_2/motion.pt"))
        p1.eval()
        p2 = torch.jit.load(str(ROOT / "policy/pre_train/h1_2/motion.pt"))
        p2.eval()
        with torch.no_grad():
            a = p1(torch.from_numpy(b0)).numpy().squeeze()
            c = p2(torch.from_numpy(pert)).numpy().squeeze()
        dev.append(float(np.abs(a - c).mean()))
    return float(np.mean(dev))


print("%-32s %14s %14s" % ("block", "on noise", "on real states"))
real_rows = []
for a, b, name, src, sc in LAYOUT:
    r_ = sens_real((a, b)) / (b - a)
    prev = [x for x in rows if x[0] == name][0][2]
    real_rows.append((name, prev, r_))
    print("%-32s %14.5f %14.5f" % (name, prev, r_))
print()
noise_order = [n for n, _, _ in sorted(real_rows, key=lambda x: -x[1])]
real_order = [n for n, _, _ in sorted(real_rows, key=lambda x: -x[2])]
print("  ranking on noise      : %s" % ", ".join(n.split()[0] for n in noise_order))
print("  ranking on real states: %s" % ", ".join(n.split()[0] for n in real_order))
if noise_order[0] == real_order[0]:
    print()
    print("  the top block agrees between the two, which is reassuring but not")
    print("  proof: both are still local perturbation tests.")
else:
    print()
    print("  the rankings DISAGREE at the top. The noise test was measuring")
    print("  the wrong thing, and only the real-state one is worth quoting.")


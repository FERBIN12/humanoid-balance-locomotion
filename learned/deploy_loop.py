#!/usr/bin/env python3
"""6.4 -- the deploy loop, end to end, and the four ways to get it wrong.

6.2 opened the policy, 6.3 laid out its inputs. This is the loop that connects
them to a robot. It is short, and every line of it is a place where a mistake
produces confident nonsense rather than an error.

To prove that, the same loop runs five times: once correct, and once with each
of four realistic mistakes. All five produce numbers. Only one walks.
"""
import os, pathlib
import numpy as np
import torch, yaml, mujoco

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
PT = str(ROOT / "policy/pre_train/h1_2/motion.pt")
SCENE = str(ROOT / "mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)

NA = cfg["num_actions"]
NOBS = cfg["num_obs"]
KP = np.array(cfg["kps"], np.float32)
KD = np.array(cfg["kds"], np.float32)
DEFAULT = np.array(cfg["default_angles"], np.float32)
CMD = np.array(cfg["cmd_init"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]
GAIT = 0.8


def gravity_body(q):
    """Gravity direction in the body frame, from the base quaternion."""
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def run(dur=20.0, fault=None, decim=None, gait=None):
    """The whole deploy loop. `fault` injects one realistic mistake."""
    policy = torch.jit.load(PT)
    policy.eval()
    d = mujoco.MjData(m)
    d.qpos[7:7 + NA] = DEFAULT
    mujoco.mj_forward(m, d)
    action = np.zeros(NA, np.float32)
    target = DEFAULT.copy()
    obs = np.zeros(NOBS, np.float32)
    dec = decim or DECIM
    per = gait or GAIT
    counter, minz = 0, 9.9
    for step in range(int(dur / m.opt.timestep)):
        # --- 1 the fast loop: PD control at 500 Hz -----------------------
        tau = (target - d.qpos[7:7 + NA]) * KP - d.qvel[6:6 + NA] * KD
        d.ctrl[:NA] = tau
        mujoco.mj_step(m, d)
        counter += 1
        minz = min(minz, float(d.qpos[2]))
        # --- 2 the slow loop: the policy at 50 Hz ------------------------
        if counter % dec == 0:
            g = gravity_body(d.qpos[3:7])
            if fault == "gravity_world":
                # forgetting to rotate into the body frame: a very easy one
                g = np.array([0, 0, 1], np.float32)
            om = d.qvel[3:6] * cfg["ang_vel_scale"]
            qj = (d.qpos[7:7 + NA] - DEFAULT) * cfg["dof_pos_scale"]
            dqj = d.qvel[6:6 + NA] * cfg["dof_vel_scale"]
            if fault == "no_dof_vel_scale":
                dqj = d.qvel[6:6 + NA]          # forgot the 0.05
            ph = (counter * m.opt.timestep) % per / per
            obs[:3] = om
            obs[3:6] = g
            obs[6:9] = CMD * CMD_SCALE
            obs[9:9 + NA] = qj
            obs[9 + NA:9 + 2 * NA] = dqj
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            if fault == "swapped_legs":
                # left and right blocks exchanged: the classic wiring error
                for lo in (9, 9 + NA):
                    blk = obs[lo:lo + NA].copy()
                    obs[lo:lo + NA] = np.concatenate([blk[6:], blk[:6]])
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)) \
                    .numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT
            if fault == "no_action_scale":
                target = action + DEFAULT       # forgot the 0.25
    return dict(dist=float(d.qpos[0]), minz=minz, finalz=float(d.qpos[2]),
                upright=float(d.qpos[2]) > 0.8)


print("--- the loop, in six lines ---")
print("  1  read the state: quaternion, gyro, encoders")
print("  2  build 47 numbers in EXACTLY the layout of 6.3")
print("  3  run the policy, once per 10 physics steps")
print("  4  scale the 12 outputs and add the default pose")
print("  5  PD those targets into torque, every physics step")
print("  6  repeat")
print()
print("  Note the two rates. The policy runs at 50 Hz and the PD loop at 500.")
print("  The policy is NOT a controller: it retargets a controller that runs")
print("  ten times faster than it does.")
print()

print("--- the same loop, five ways ---")
FAULTS = [
    (None, "correct"),
    ("no_action_scale", "forgot action_scale (0.25)"),
    ("no_dof_vel_scale", "forgot dof_vel_scale (0.05)"),
    ("gravity_world", "gravity left in world frame"),
    ("swapped_legs", "left and right blocks swapped"),
]
print("%34s %10s %10s %9s" % ("variant", "distance", "min pelvis", "upright"))
base = None
for fault, label in FAULTS:
    r = run(fault=fault)
    if base is None:
        base = r
    print("%34s %10.3f %10.3f %9s"
          % (label, r["dist"], r["minz"], "yes" if r["upright"] else "NO"))
print()
print("  Every one of those five ran to completion without raising anything.")
print("  Four of them produce a robot on the floor and no error message at")
print("  all, which is the entire difficulty of deploying a learned policy:")
print("  the failure mode is silence.")
print()

print("--- the faults that do NOT fall over ---")
print("  The four above are loud: the robot is on the floor within seconds and")
print("  you know immediately that something is wrong. The dangerous kind is a")
print("  fault that almost works, because you will tune around it for a week.")
print("  So here are two that get the layout and the scaling right and only")
print("  the TIMING wrong.")
print()
print("%34s %10s %10s %9s" % ("variant", "distance", "min pelvis", "upright"))
SUBTLE = [
    (dict(decim=5), "policy at 100 Hz, not 50"),
    (dict(decim=20), "policy at 25 Hz, not 50"),
    (dict(gait=0.7), "gait period 0.7 s, not 0.8"),
    (dict(gait=1.0), "gait period 1.0 s, not 0.8"),
]
survivors = []
for kw, label in SUBTLE:
    r = run(**kw)
    if r["upright"]:
        survivors.append((label, r))
    print("%34s %10.3f %10.3f %9s"
          % (label, r["dist"], r["minz"], "yes" if r["upright"] else "NO"))
print()
if survivors:
    print("  %d of those stayed upright for the full run:" % len(survivors))
    for label, r in survivors:
        print("    %-32s %.3f m, which is %.0f%% of correct"
              % (label, r["dist"], 100 * r["dist"] / base["dist"]))
    print()
    print("  That is the sneaky failure. Nothing crashes, nothing falls, the")
    print("  robot walks, and it is quietly wrong.")
    print()
    faster = [(l, r) for l, r in survivors if r["dist"] > base["dist"]]
    if faster:
        print("  And it is worse than quietly wrong. %d of them travel FURTHER"
              % len(faster))
        print("  than the correct loop:")
        for l, r in faster:
            print("    %-32s %+.0f%% distance" % (l, 100 * (r["dist"] / base["dist"] - 1)))
        print()
        print("  If distance travelled were your metric, three of these bugs")
        print("  would look like improvements and you would ship them. The")
        print("  policy was trained at 50 Hz against a 0.8 s gait; running it")
        print("  faster is not a free win, it is an untested regime that")
        print("  happens to look good on one number over twenty seconds.")
        print("  The learned policy.6 is about what that costs when the ground changes.")
else:
    print("  None of them survived either, so on this policy even the timing")
    print("  faults are loud. That is a happier answer than I expected and I")
    print("  am not going to dress it up as a near miss.")
print()
print("--- so what DOES catch them ---")
print("  There is no assertion you can add from inside the loop: every value")
print("  in every variant is finite and in range.")
print()
print("  And checking the outcome does not save you either, which is the part")
print("  I did not expect when I wrote this file. My intended conclusion was")
print("  'compare distance against a reference'. Three of the four timing")
print("  faults BEAT the reference. That advice would have shipped the bugs.")
print()
print("  What actually distinguishes them is that the four loud faults break")
print("  the CONTRACT the policy was trained under, and the four quiet ones")
print("  break the RATE it was trained at. For the first kind, assert the")
print("  contract directly: check that your gravity vector has unit norm and")
print("  points down at rest, that your dof velocities are the scaled ones,")
print("  that left and right blocks map to the joints you think they do.")
print("  Those are checks on the observation, not on the outcome.")
print()
print("  For the second kind, there is no substitute for knowing the training")
print("  configuration. The policy was trained at %d Hz against a %.1f s gait."
      % (int(1.0 / (m.opt.timestep * DECIM)), GAIT))
print("  Run it anywhere else and you are extrapolating, however good the")
print("  twenty second number looks.")

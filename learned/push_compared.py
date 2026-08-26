#!/usr/bin/env python3
"""6.7 -- push both controllers and watch HOW each one fails.

6.6 established the numbers: the classical stack survives 250.23 N, the policy
450. That is one number each and it hides the interesting part, which is that
the two fail in completely different ways.

Both controllers, same robot, same push, same instant.
"""
import os, pathlib
import numpy as np
import torch, yaml, mujoco

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
PT = str(ROOT / "policy/pre_train/h1_2/motion.pt")
SCENE = str(ROOT / "mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
NA, NOBS = cfg["num_actions"], cfg["num_obs"]
KPP = np.array(cfg["kps"], np.float32)
KDP = np.array(cfg["kds"], np.float32)
DEFAULT = np.array(cfg["default_angles"], np.float32)
CMD = np.array(cfg["cmd_init"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]
FL = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
FR = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
OMEGA = np.sqrt(9.81 / 0.937)

# the classical stack: the x10 gains from 3.4, holding the crouch
KPC = np.array([200., 200., 200., 300., 60., 40.] * 2) * 10.0
KDC = np.array([5., 5., 5., 7.5, 2., 2.] * 2) * np.sqrt(10.0)
NAMES = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
         for i in range(m.nu)]
IDX = {n: i for i, n in enumerate(NAMES)}
QA = np.array([m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)])
VA = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)])


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def trace(kind, push_N, t_push=6.0, dur=12.0):
    """Run one controller through one push, logging the response."""
    d = mujoco.MjData(m)
    policy = None
    if kind == "learned":
        policy = torch.jit.load(PT)
        policy.eval()
        d.qpos[7:7 + NA] = DEFAULT
    mujoco.mj_forward(m, d)
    action = np.zeros(NA, np.float32)
    target = DEFAULT.copy()
    obs = np.zeros(NOBS, np.float32)
    counter = 0
    log = []
    fell = None
    for step in range(int(dur / m.opt.timestep)):
        t = step * m.opt.timestep
        d.xfrc_applied[1][0] = push_N if t_push <= t < t_push + 0.2 else 0.0
        if kind == "learned":
            d.ctrl[:NA] = (target - d.qpos[7:7 + NA]) * KPP \
                - d.qvel[6:6 + NA] * KDP
        else:
            d.ctrl[:] = KPC * (0 - d.qpos[QA]) - KDC * d.qvel[VA]
        mujoco.mj_step(m, d)
        counter += 1
        if kind == "learned" and counter % DECIM == 0:
            ph = (counter * m.opt.timestep) % 0.8 / 0.8
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_body(d.qpos[3:7])
            obs[6:9] = CMD * CMD_SCALE
            obs[9:9 + NA] = (d.qpos[7:7 + NA] - DEFAULT) * cfg["dof_pos_scale"]
            obs[9 + NA:9 + 2 * NA] = d.qvel[6:6 + NA] * cfg["dof_vel_scale"]
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)) \
                    .numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT
        if fell is None and float(d.qpos[2]) < 0.55:
            fell = t
        if step % 10 == 0:
            mujoco.mj_subtreeVel(m, d)
            mid = (d.xpos[FL] + d.xpos[FR]) / 2.0
            com = d.subtree_com[0]
            cap = float(com[0] - mid[0]) + float(d.subtree_linvel[0][0]) / OMEGA
            log.append((t, float(d.qpos[2]), float(com[0] - mid[0]), cap,
                        int(d.ncon)))
    return np.array(log), fell


print("--- 1 the same push, both controllers ---")
PUSH = 200.0
print("  %.0f N for 200 ms at t=6.0 s, which BOTH survive" % PUSH)
print()
print("%10s %14s %14s %14s"
      % ("t (s)", "classical com", "learned com", "learned ncon"))
lc, fc = trace("classical", PUSH)
ll, fl = trace("learned", PUSH)
for tt in (5.8, 6.0, 6.2, 6.5, 7.0, 8.0, 10.0):
    ic = int(np.argmin(np.abs(lc[:, 0] - tt)))
    il = int(np.argmin(np.abs(ll[:, 0] - tt)))
    print("%10.1f %14.3f %14.3f %14d"
          % (tt, lc[ic, 2], ll[il, 2], ll[il, 4]))
print()
print("  Both are upright at the end. The classical stack's CoM offset peaks")
print("  at %.3f m and settles back to %.3f. The learned policy peaks at %.3f"
      % (np.abs(lc[:, 2]).max(), lc[-1, 2], np.abs(ll[:, 2]).max()))
print("  and is at %.3f: it does not settle back, because it is walking away."
      % ll[-1, 2])
print()

print("--- 2 how each one uses its feet ---")
after = ll[ll[:, 0] > 6.0]
print("  learned, after the push: contact count min %d, max %d"
      % (after[:, 4].min(), after[:, 4].max()))
ac = lc[lc[:, 0] > 6.0]
print("  classical, after the push: contact count min %d, max %d"
      % (ac[:, 4].min(), ac[:, 4].max()))
print()
print("  The classical stack keeps both feet planted: it is an ankle strategy")
print("  and 3.8 built it that way. The learned policy takes steps.")
print()

print("--- 3 the capture point, which 4.5 said bounds everything ---")
print("%10s %16s %16s" % ("t (s)", "classical cap", "learned cap"))
for tt in (5.8, 6.2, 6.5, 7.0, 8.0):
    ic = int(np.argmin(np.abs(lc[:, 0] - tt)))
    il = int(np.argmin(np.abs(ll[:, 0] - tt)))
    print("%10.1f %16.3f %16.3f" % (tt, lc[ic, 3], ll[il, 3]))
print()
print("  4.5 measured the reachable step at 0.493 m. Watch whether either")
print("  controller's capture point goes past it.")
print("  classical peak %.3f m, learned peak %.3f m"
      % (np.abs(lc[:, 3]).max(), np.abs(ll[:, 3]).max()))
print()

print("--- 4 at the classical limit ---")
for N in (250.0, 260.0):
    _, f1 = trace("classical", N)
    _, f2 = trace("learned", N)
    print("  %.0f N: classical %s, learned %s"
          % (N, ("fell at %.2f s" % f1) if f1 else "survived",
             ("fell at %.2f s" % f2) if f2 else "survived"))
print()
print("--- 5 at the learned limit ---")
for N in (450.0, 460.0):
    _, f2 = trace("learned", N)
    print("  %.0f N: learned %s"
          % (N, ("fell at %.2f s" % f2) if f2 else "survived"))

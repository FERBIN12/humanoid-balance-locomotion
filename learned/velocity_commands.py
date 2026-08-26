#!/usr/bin/env python3
"""6.5 -- the three numbers an operator actually controls.

Of the 47 inputs, exactly three come from a human: forward speed, sideways
speed, and turn rate. This measures what the policy does with them, which is
not the same as what you asked for.

Every row is a separate 20 s run in MuJoCo.
"""
import os, pathlib
import numpy as np
import torch, yaml, mujoco

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
PT = str(ROOT / "policy/pre_train/h1_2/motion.pt")
m = mujoco.MjModel.from_xml_path(
    str(ROOT / "mujoco/resources/robots/h1_2/scene.xml"))
NA, NOBS = cfg["num_actions"], cfg["num_obs"]
KP = np.array(cfg["kps"], np.float32)
KD = np.array(cfg["kds"], np.float32)
DEFAULT = np.array(cfg["default_angles"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def yaw_of(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def run(cmd, dur=20.0):
    policy = torch.jit.load(PT)
    policy.eval()
    d = mujoco.MjData(m)
    d.qpos[7:7 + NA] = DEFAULT
    mujoco.mj_forward(m, d)
    action = np.zeros(NA, np.float32)
    target = DEFAULT.copy()
    obs = np.zeros(NOBS, np.float32)
    c = np.array(cmd, np.float32)
    counter, minz = 0, 9.9
    # Accumulate PATH LENGTH, not straight line displacement. A robot that
    # walks in a circle ends up near where it started, and reading its
    # displacement says it barely moved. Measured: a "forward + turn" run
    # traces a 1.07 m radius arc, covering 7.34 m at 0.367 m/s, while its
    # displacement is 0.604 m. Those are the same run.
    path = 0.0
    px, py = float(d.qpos[0]), float(d.qpos[1])
    yaw0 = yaw_of(d.qpos[3:7])
    unwrapped, prev = 0.0, yaw0
    for step in range(int(dur / m.opt.timestep)):
        d.ctrl[:NA] = (target - d.qpos[7:7 + NA]) * KP - d.qvel[6:6 + NA] * KD
        mujoco.mj_step(m, d)
        counter += 1
        minz = min(minz, float(d.qpos[2]))
        cx, cy = float(d.qpos[0]), float(d.qpos[1])
        path += float(np.hypot(cx - px, cy - py))
        px, py = cx, cy
        yw = yaw_of(d.qpos[3:7])
        dy = yw - prev
        if dy > np.pi:
            dy -= 2 * np.pi
        elif dy < -np.pi:
            dy += 2 * np.pi
        unwrapped += dy
        prev = yw
        if counter % DECIM == 0:
            ph = (counter * m.opt.timestep) % 0.8 / 0.8
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_body(d.qpos[3:7])
            obs[6:9] = c * CMD_SCALE
            obs[9:9 + NA] = (d.qpos[7:7 + NA] - DEFAULT) * cfg["dof_pos_scale"]
            obs[9 + NA:9 + 2 * NA] = d.qvel[6:6 + NA] * cfg["dof_vel_scale"]
            obs[9 + 2 * NA:9 + 3 * NA] = action
            obs[9 + 3 * NA] = np.sin(2 * np.pi * ph)
            obs[9 + 3 * NA + 1] = np.cos(2 * np.pi * ph)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs).unsqueeze(0)) \
                    .numpy().squeeze()
            target = action * cfg["action_scale"] + DEFAULT
    return dict(x=float(d.qpos[0]), y=float(d.qpos[1]), yaw=unwrapped,
                vx=float(d.qpos[0]) / dur, vy=float(d.qpos[1]) / dur,
                wz=unwrapped / dur, minz=minz, path=path, speed=path / dur,
                upright=float(d.qpos[2]) > 0.8)


print("--- 1 forward: does it go the speed you asked for ---")
print("%10s %12s %12s %10s %9s"
      % ("cmd vx", "achieved", "error", "min pelvis", "upright"))
fwd = []
for v in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5):
    r = run([v, 0.0, 0.0])
    fwd.append((v, r))
    print("%10.2f %12.3f %+12.3f %10.3f %9s"
          % (v, r["vx"], r["vx"] - v, r["minz"],
             "yes" if r["upright"] else "NO"))
print()
ok = [(v, r) for v, r in fwd if r["upright"]]
if ok:
    errs = [abs(r["vx"] - v) for v, r in ok]
    print("  across the runs that stayed up, the speed error is %.3f to %.3f"
          % (min(errs), max(errs)))
    print("  m/s. The command is a REQUEST, not a setpoint: there is no")
    print("  integrator anywhere in this system to remove that error.")
print()

print("--- 2 sideways: the direction the walking stack could not manage at all ---")
print("%10s %12s %12s %12s %9s"
      % ("cmd vy", "achieved vy", "drift x", "min pelvis", "upright"))
for v in (0.0, 0.2, 0.4):
    r = run([0.0, v, 0.0])
    print("%10.2f %12.3f %12.3f %12.3f %9s"
          % (v, r["vy"], r["x"], r["minz"],
             "yes" if r["upright"] else "NO"))
print()

print("--- 3 turning ---")
print("%10s %12s %12s %10s %9s"
      % ("cmd wz", "achieved", "total yaw", "min pelvis", "upright"))
for wz in (0.0, 0.3, 0.6):
    r = run([0.0, 0.0, wz])
    print("%10.2f %12.3f %12.2f %10.3f %9s"
          % (wz, r["wz"], r["yaw"], r["minz"],
             "yes" if r["upright"] else "NO"))
print()

print("--- 4 combined, and a metric that nearly fooled me ---")
print("  Reading straight line displacement, 'forward + turn' looks like it")
print("  almost stops: 0.03 m/s against 0.46 forward only. That is wrong, and")
print("  it is wrong in a way worth showing. A robot turning while walking")
print("  traces a CIRCLE, so it ends up near where it started however far it")
print("  actually travelled. Path length is the honest measure.")
print()
print("%22s %11s %11s %10s %9s"
      % ("command", "displacement", "path length", "turn rate", "upright"))
for cmd, label in (([0.5, 0.0, 0.0], "forward only"),
                   ([0.5, 0.2, 0.0], "forward + sideways"),
                   ([0.5, 0.0, 0.4], "forward + turn"),
                   ([0.5, 0.2, 0.4], "all three")):
    r = run(cmd)
    print("%22s %11.3f %11.3f %10.3f %9s"
          % (label, float(np.hypot(r["x"], r["y"])), r["path"], r["wz"],
             "yes" if r["upright"] else "NO"))
print()
rt = run([0.5, 0.0, 0.4])
print("  'forward + turn' covers %.2f m of ground at %.3f m/s along its path,"
      % (rt["path"], rt["speed"]))
print("  ending %.3f m from where it started. Both numbers are true. Only one"
      % float(np.hypot(rt["x"], rt["y"])))
print("  of them describes what the robot did.")

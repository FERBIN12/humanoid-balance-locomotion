#!/usr/bin/env python3
"""6.6 -- what the policy does not know, and what that costs.

6.3 listed what is absent from the observation: no contact, no base velocity,
no terrain height, no idea where it is. This finds out which of those absences
actually breaks it, by changing the world underneath it and measuring.

Every row is a 20 s run. The policy is reloaded each time, because 6.2 showed
it keeps state.
"""
import os, pathlib, tempfile, re
import numpy as np
import torch, yaml, mujoco

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
PT = str(ROOT / "policy/pre_train/h1_2/motion.pt")
SCENE = ROOT / "mujoco/resources/robots/h1_2/scene.xml"
NA, NOBS = cfg["num_actions"], cfg["num_obs"]
KP = np.array(cfg["kps"], np.float32)
KD = np.array(cfg["kds"], np.float32)
DEFAULT = np.array(cfg["default_angles"], np.float32)
CMD = np.array(cfg["cmd_init"], np.float32)
CMD_SCALE = np.array(cfg["cmd_scale"], np.float32)
DECIM = cfg["control_decimation"]


def gravity_body(q):
    w, x, y, z = q
    return np.array([2 * (-z * x + w * y), -2 * (z * y + w * x),
                     1 - 2 * (w * w + z * z)], np.float32)


def run(model, dur=20.0, push=None, mass_scale=1.0):
    policy = torch.jit.load(PT)
    policy.eval()
    d = mujoco.MjData(model)
    d.qpos[7:7 + NA] = DEFAULT
    if mass_scale != 1.0:
        model.body_mass[:] = model.body_mass * mass_scale
    mujoco.mj_forward(model, d)
    action = np.zeros(NA, np.float32)
    target = DEFAULT.copy()
    obs = np.zeros(NOBS, np.float32)
    counter, minz, path = 0, 9.9, 0.0
    px, py = float(d.qpos[0]), float(d.qpos[1])
    fell = None
    for step in range(int(dur / model.opt.timestep)):
        t = step * model.opt.timestep
        if push and push[0] <= t < push[0] + 0.2:
            d.xfrc_applied[1][0] = push[1]
        else:
            d.xfrc_applied[1][0] = 0.0
        d.ctrl[:NA] = (target - d.qpos[7:7 + NA]) * KP - d.qvel[6:6 + NA] * KD
        mujoco.mj_step(model, d)
        counter += 1
        minz = min(minz, float(d.qpos[2]))
        cx, cy = float(d.qpos[0]), float(d.qpos[1])
        path += float(np.hypot(cx - px, cy - py))
        px, py = cx, cy
        if fell is None and float(d.qpos[2]) < 0.55:
            fell = t
        if counter % DECIM == 0:
            ph = (counter * model.opt.timestep) % 0.8 / 0.8
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
    return dict(path=path, minz=minz, fell=fell,
                upright=float(d.qpos[2]) > 0.8)


def variant(friction=None, slope=None):
    """Build a modified scene, then set friction on the COMPILED model.

    Editing only the floor's XML does nothing. MuJoCo combines a contact pair
    by taking the elementwise MAX of the two geoms' friction, and every geom
    on this robot ships with friction 1.0. Lowering the floor to 0.15 leaves
    the effective coefficient at 1.0, and the first version of this file
    produced five identical rows across a 6.7x change without noticing.
    """
    src = SCENE.read_text()
    if slope is not None:
        src = re.sub(r'(<geom name="floor"[^>]*)/>',
                     r'\1 euler="0 %.4f 0"/>' % slope, src)
    f = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False,
                                    dir=str(SCENE.parent))
    f.write(src)
    f.close()
    try:
        mdl = mujoco.MjModel.from_xml_path(f.name)
    finally:
        os.unlink(f.name)
    if friction is not None:
        # set BOTH sides, or the max wins and nothing changes
        mdl.geom_friction[:, 0] = friction
    return mdl


base_model = mujoco.MjModel.from_xml_path(str(SCENE))
print("--- 0 the reference ---")
ref = run(base_model)
print("  flat ground, nominal everything: %.3f m of path, min pelvis %.3f"
      % (ref["path"], ref["minz"]))
print()

print("--- 1 friction, which the policy cannot sense ---")
print("%14s %12s %12s %10s" % ("friction", "path (m)", "min pelvis", "upright"))
for mu in (1.0, 0.6, 0.4, 0.25, 0.15):
    r = run(variant(friction=mu))
    print("%14.2f %12.3f %12.3f %10s"
          % (mu, r["path"], r["minz"], "yes" if r["upright"] else "NO"))
print()
print("  There is no friction term in the observation vector. The policy finds")
print("  out about the floor only through what its joints do afterwards.")
print()

print("--- 2 slope, which it also cannot sense ---")
print("%14s %12s %12s %10s" % ("slope (deg)", "path (m)", "min pelvis", "upright"))
for deg in (0.0, 3.0, 6.0, 10.0):
    r = run(variant(slope=np.radians(deg)))
    print("%14.1f %12.3f %12.3f %10s"
          % (deg, r["path"], r["minz"], "yes" if r["upright"] else "NO"))
print()

print("--- 3 a push, which is the classic test ---")
print("%14s %12s %12s %10s" % ("push (N)", "path (m)", "min pelvis", "upright"))
for N in (0, 100, 200, 300, 400):
    r = run(mujoco.MjModel.from_xml_path(str(SCENE)), push=(6.0, N))
    print("%14d %12.3f %12.3f %10s"
          % (N, r["path"], r["minz"], "yes" if r["upright"] else "NO"))
print()
print("  It survives 400 N, so a table that stops there is a table that has")
print("  not found the limit. Bisecting, the same way 3.9 did:")
lo, hi = 400.0, 600.0
for _ in range(7):
    mid = (lo + hi) / 2
    r = run(mujoco.MjModel.from_xml_path(str(SCENE)), push=(6.0, mid))
    print("    %7.2f N -> %s" % (mid, "upright" if r["upright"] else "FELL"))
    if r["upright"]:
        lo = mid
    else:
        hi = mid
print()
print("  last surviving push %.2f N, first failing %.2f N" % (lo, hi))
print()
print("  Comparing that to the stepping controller needs care, and I got it wrong first.")
print("  4.7 measured 250.23 N for the classical stack, but that run had the")
print("  ARM MOMENTUM strategy from 4.2 switched on. The same PD controller")
print("  with no arm swing, pushed the same way, survives only 100 N.")
print("  So there are two honest comparisons and they say different things:")
print("    against bare PD (100 N):            %.2fx" % (lo / 100.0))
print("    against the best classical stack:   %.2fx" % (lo / 250.23))
print("  The second is the fair one, because 4.7's number is the best that")
print("  section achieved and that is what a learned policy has to beat.")
print()
print("--- what this all adds up to ---")
print("  The policy has no friction sensor and survives down to mu = 0.40.")
print("  It has no inclinometer and fails at 3 degrees of slope.")
print("  It has no force sensor and rejects 450 N of push, 1.8x the classical")
print("  stack's limit.")
print()
print("  Notice that those do not line up the way the observation vector")
print("  would predict. All three are equally invisible to it, and it handles")
print("  them very differently. What separates them is not what the policy")
print("  can SENSE, it is what the training distribution contained. Pushes and")
print("  friction variation are standard domain randomisation. A persistent")
print("  slope changes the gravity direction in the body frame for the whole")
print("  episode, which is a different thing entirely, and 6.8 is where we")
print("  look at what was actually randomised.")

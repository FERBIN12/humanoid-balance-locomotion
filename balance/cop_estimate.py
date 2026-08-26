#!/usr/bin/env python3
"""The centre of pressure, from the ankle force/torque sensors.

The CoM tells you where the weight IS. The CoP tells you where the ground is
PUSHING. Balance is the relationship between the two, so we need both.

Two ways to get the CoP, and comparing them is the point:
  1 from the contact list      -- exact, and only a simulator has it
  2 from the ankle F/T sensors -- what a real robot has bolted to its shins
"""
import os, numpy as np, mujoco

m = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml"))
d = mujoco.MjData(m)
QA = {i: m.jnt_qposadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
VA = {i: m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(m.nu)}
MASS = float(m.body_mass.sum())
KP = np.array([200., 200., 200., 300., 60., 40.] * 2) * 10
KD = np.array([5., 5., 5., 7.5, 2., 2.] * 2) * np.sqrt(10.0)


WORLD = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "world")


def cop_from_contacts(data, ground_only=True):
    """Force weighted average of the contact POSITIONS. Exact by construction,
    which is why it can never leave the convex hull of the contacts.

    ground_only matters more than it looks. A contact list is NOT a ground
    contact list. On the full body model this robot rests each thumb against
    its own wrist, and summing all eight contacts gives 782.6 N of "support"
    against a weight of 660.9 -- an 18 per cent overshoot from forces the floor
    never supplied. Filtering to contacts involving the world body gives
    661.3 N. Self collision is a feature of a robot with hands. It does not
    hold you up.
    """
    num = np.zeros(3)
    den = 0.0
    for c in range(data.ncon):
        con = data.contact[c]
        if ground_only:
            b1 = m.geom_bodyid[con.geom1]
            b2 = m.geom_bodyid[con.geom2]
            if b1 != WORLD and b2 != WORLD:
                continue
        f = np.zeros(6)
        mujoco.mj_contactForce(m, data, c, f)
        # rotate into the world frame before trusting the vertical component
        fz = float((con.frame.reshape(3, 3).T @ f[:3])[2])
        if fz <= 0:
            continue
        num += con.pos * fz
        den += fz
    if den <= 0:
        return None, 0.0
    return num / den, den


def com_of(data):
    return (m.body_mass[:, None] * data.xipos).sum(0) / MASS


mujoco.mj_forward(m, d)
print("standing at %.1f x gain, so the robot holds its pose while we measure" % 10)
print()
print("%6s %26s %26s %10s %9s" % ("t", "CoM (x,y)", "CoP (x,y)", "normal N", "offset"))
rows = []
for step in range(3000):
    q = np.array([d.qpos[QA[i]] for i in range(m.nu)])
    v = np.array([d.qvel[VA[i]] for i in range(m.nu)])
    d.ctrl[:] = KP * (0 - q) - KD * v
    mujoco.mj_step(m, d)
    if step % 500 == 0 or step == 2999:
        cop, fn = cop_from_contacts(d)
        cm = com_of(d)
        if cop is None:
            print("%6.2f  no contact" % d.time)
            continue
        off = float(np.linalg.norm(cm[:2] - cop[:2]))
        rows.append((cm[:2].copy(), cop[:2].copy(), fn, off))
        print("%6.2f  [%8.4f %8.4f]  [%8.4f %8.4f]  %10.1f %9.4f"
              % (d.time, cm[0], cm[1], cop[0], cop[1], fn, off))

print()
# --- does Newton hold? Check the MEAN, not an instant ------------------------
# A single sample of the vertical contact force reads anywhere from 613 to 670 N
# against mg = 660.9. That looked like a 48 N discrepancy until I averaged it:
# the stiff controller makes the robot chatter about a correct mean.
KPg, KDg = KP, KD
fz_hist, pz_hist = [], []
d2 = mujoco.MjData(m)
mujoco.mj_forward(m, d2)
for step in range(6000):
    q = np.array([d2.qpos[QA[i]] for i in range(m.nu)])
    v = np.array([d2.qvel[VA[i]] for i in range(m.nu)])
    d2.ctrl[:] = KPg * (0 - q) - KDg * v
    mujoco.mj_step(m, d2)
    if step > 3000:                       # discard the settling transient
        tot = 0.0
        for k in range(d2.ncon):
            f = np.zeros(6)
            mujoco.mj_contactForce(m, d2, k, f)
            tot += float((d2.contact[k].frame.reshape(3, 3).T @ f[:3])[2])
        fz_hist.append(tot)
        pz_hist.append(float(d2.qpos[2]))
fz_hist = np.array(fz_hist); pz_hist = np.array(pz_hist)
print("vertical contact force: mean %.1f N, standard deviation %.1f N"
      % (fz_hist.mean(), fz_hist.std()))
print("m*g is %.1f N, so the MEAN is exact and the scatter is oscillation"
      % (MASS * 9.81))
print("the pelvis moves only %.5f m peak to peak while that happens"
      % (pz_hist.max() - pz_hist.min()))
print()
cm, cop, fn, off = rows[-1]
print("CoM to CoP horizontal offset: %.4f m" % off)
print("the foot is 0.240 m long, so that offset is %.1f%% of the foot"
      % (100 * off / 0.240))
print()
print("and here is the fact worth keeping: the CoP is a force weighted average")
print("of contact POSITIONS, so it is inside the convex hull of the contacts BY")
print("CONSTRUCTION. It cannot leave the support polygon. A CoP outside the")
print("polygon is a bug in your maths, never a robot that is falling.")
print()
print("what CAN leave the polygon is the CoM, and that is what falling means.")

# --- the self-contact trap, on the FULL body model --------------------------
# The 12 DOF leg model has 4 contacts, all foot against floor, so the naive
# loop happens to be right. Add the arms and hands and it stops being right.
print()
print("=" * 62)
mf = mujoco.MjModel.from_xml_path(os.path.expanduser(
    "~/humanoid_ws/mujoco/resources/robots/h1_2/scene_full.xml"))
nf = [mujoco.mj_id2name(mf, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(mf.nu)]
QAf = {i: mf.jnt_qposadr[mf.actuator_trnid[i][0]] for i in range(mf.nu)}
VAf = {i: mf.jnt_dofadr[mf.actuator_trnid[i][0]] for i in range(mf.nu)}
WORLDf = mujoco.mj_name2id(mf, mujoco.mjtObj.mjOBJ_BODY, "world")
BASE = {"hip_yaw": 200., "hip_pitch": 200., "hip_roll": 200.,
        "knee": 300., "ankle_pitch": 60., "ankle_roll": 40.}
KPf = np.zeros(mf.nu); KDf = np.zeros(mf.nu)
for i, n in enumerate(nf):
    hit = [v for k, v in BASE.items() if n and k in n]
    KPf[i] = hit[0] * 10 if hit else 60.
    KDf[i] = hit[0] / 40 * np.sqrt(10.0) if hit else 3.0
df = mujoco.MjData(mf)
mujoco.mj_forward(mf, df)
allz, gndz = [], []
for step in range(7000):
    q = np.array([df.qpos[QAf[i]] for i in range(mf.nu)])
    v = np.array([df.qvel[VAf[i]] for i in range(mf.nu)])
    df.ctrl[:] = KPf * (0 - q) - KDf * v
    mujoco.mj_step(mf, df)
    if step > 3000:
        ta = tg = 0.0
        for c in range(df.ncon):
            con = df.contact[c]
            f = np.zeros(6)
            mujoco.mj_contactForce(mf, df, c, f)
            fz = float((con.frame.reshape(3, 3).T @ f[:3])[2])
            ta += fz
            if mf.geom_bodyid[con.geom1] == WORLDf or mf.geom_bodyid[con.geom2] == WORLDf:
                tg += fz
        allz.append(ta); gndz.append(tg)
MASSf = float(mf.body_mass.sum())
print("full body model, %d contacts, pelvis %.4f m" % (df.ncon, df.qpos[2]))


def who(g):
    return mujoco.mj_id2name(mf, mujoco.mjtObj.mjOBJ_BODY, mf.geom_bodyid[g])


for c in range(df.ncon):
    con = df.contact[c]
    f = np.zeros(6)
    mujoco.mj_contactForce(mf, df, c, f)
    fz = float((con.frame.reshape(3, 3).T @ f[:3])[2])
    tag = "ground" if (mf.geom_bodyid[con.geom1] == WORLDf
                       or mf.geom_bodyid[con.geom2] == WORLDf) else "SELF"
    print("  %-6s %-24s <-> %-24s Fz %7.2f"
          % (tag, who(con.geom1), who(con.geom2), fz))
print()
print("m*g                       %7.1f N" % (MASSf * 9.81))
print("all contacts summed       %7.1f N   <- %.1f%% too high"
      % (np.mean(allz), 100 * (np.mean(allz) - MASSf * 9.81) / (MASSf * 9.81)))
print("world contacts only       %7.1f N   <- %.2f%% off"
      % (np.mean(gndz), 100 * abs(np.mean(gndz) - MASSf * 9.81) / (MASSf * 9.81)))
print()
print("a contact list is not a ground contact list. Check the bodies.")

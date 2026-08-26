#!/usr/bin/env python3
"""5.6 -- turning a desired motion into joint torques.

5.5 gave us joint ANGLES for the swing leg. A torque controlled robot cannot
use angles directly: it needs torques, and the torques that produce one joint's
motion depend on every other joint, on gravity, and on what the feet are doing.

That coupling is what "whole body" means, and the standard tool is a quadratic
program. There is no QP solver installed here on purpose. An equality
constrained QP has a closed form solution through its KKT system, and building
it from numpy shows the mechanism instead of hiding it behind a library call.
"""
import numpy as np
import mujoco, os

SCENE = os.path.expanduser("~/humanoid_ws/mujoco/resources/robots/h1_2/scene.xml")
m = mujoco.MjModel.from_xml_path(SCENE)
d = mujoco.MjData(m)
NV, NU = m.nv, m.nu
NAMES = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(NU)]
IDX = {n: i for i, n in enumerate(NAMES)}
VA = np.array([m.jnt_dofadr[m.actuator_trnid[i][0]] for i in range(NU)])

print("--- 1 what the dynamics actually say ---")
mujoco.mj_forward(m, d)
print("  %d degrees of freedom, %d actuators" % (NV, NU))
print("  the first 6 dof are the floating base, and NOTHING drives them")
print("  directly. That is the whole difficulty: you steer the base by")
print("  pushing on the floor through the legs.")
print()

# M qddot + h = S tau + J^T f
M = np.zeros((NV, NV))
mujoco.mj_fullM(m, d, M)
h = d.qfrc_bias.copy()
print("  mass matrix %dx%d, condition number %.1f" % (NV, NV, np.linalg.cond(M)))
print("  bias (gravity + coriolis) norm %.2f Nm" % np.linalg.norm(h))
print()

print("--- 2 the naive answer, and why it is wrong ---")
# ask for a joint acceleration and invert the joint block alone
qdd_want = np.zeros(NV)
qdd_want[VA[IDX["left_knee_joint"]]] = 2.0        # rad/s^2
S = np.zeros((NV, NU))
for i in range(NU):
    S[VA[i], i] = 1.0
tau_naive = (M @ qdd_want + h)[VA]
print("  ask for 2.0 rad/s2 at the left knee, ignore the base:")
print("    naive tau at the knee: %.2f Nm" % tau_naive[IDX["left_knee_joint"]])
# Apply it and see what the base does. Measure the DIFFERENCE against a
# zero torque run: at this instant the robot is not yet in contact, so the
# raw base acceleration is dominated by -9.807 m/s2 of gravity and reading
# it directly would credit the knee with something gravity did.
def base_acc(tau_knee):
    dd = mujoco.MjData(m)
    mujoco.mj_forward(m, dd)
    dd.ctrl[:] = 0.0
    dd.ctrl[IDX["left_knee_joint"]] = tau_knee
    mujoco.mj_step(m, dd)
    return dd.qacc[:6].copy()


BIG = 20.0
base0 = base_acc(0.0)
base1 = base_acc(BIG)
delta = base1 - base0
print("    baseline, zero torque:  %s" % np.round(base0, 4))
print("    with %.0f Nm at the knee: %s" % (BIG, np.round(base1, 4)))
print("    the DIFFERENCE the torque caused: %s" % np.round(delta, 4))
print("      linear  %s m/s2" % np.round(delta[:3], 3))
print("      angular %s rad/s2" % np.round(delta[3:], 3))
print("  norm %.3f. The base moves, and nothing asked it to. Every joint"
      % np.linalg.norm(delta))
print("  torque is also a force on the body it is attached to.")
print()

print("--- 3 the QP, built from its KKT system ---")
print("  minimise  ||qddot - qddot_desired||^2  subject to the dynamics")
print("  being satisfiable by the actuators we actually have.")
print()


def solve_qp(qdd_des, wt_base=50.0):
    """Closed form equality constrained QP.

    Variables: x = [qddot (NV); tau (NU)]
    Cost:      || W (qddot - qdd_des) ||^2 + eps ||tau||^2
    Constraint: M qddot + h = S tau        (the equations of motion)

    KKT:  [ 2H  A^T ] [x     ]   [ 2H x_ref ]
          [ A   0   ] [lambda] = [ -h       ]
    """
    n = NV + NU
    W = np.ones(NV)
    W[:6] = wt_base            # care MORE about the base doing what we asked
    H = np.zeros((n, n))
    H[:NV, :NV] = np.diag(W)
    H[NV:, NV:] = np.eye(NU) * 1e-4      # regularise tau
    xref = np.zeros(n)
    xref[:NV] = qdd_des
    A = np.hstack([M, -S])
    b = -h
    KKT = np.block([[2 * H, A.T],
                    [A, np.zeros((NV, NV))]])
    rhs = np.concatenate([2 * H @ xref, b])
    sol = np.linalg.solve(KKT, rhs)
    return sol[:NV], sol[NV:NV + NU]


qdd_qp, tau_qp = solve_qp(qdd_want)
print("  same request, solved as a QP:")
print("    knee torque      %.2f Nm  (naive gave %.2f)"
      % (tau_qp[IDX["left_knee_joint"]], tau_naive[IDX["left_knee_joint"]]))
print("    knee accel got   %.4f rad/s2  (asked 2.0)"
      % qdd_qp[VA[IDX["left_knee_joint"]]])
print("    base accel norm  %.4f  (the QP was told to keep this small)"
      % np.linalg.norm(qdd_qp[:6]))
print()
resid = M @ qdd_qp + h - S @ tau_qp
print("  dynamics residual ||M qddot + h - S tau|| = %.3e" % np.linalg.norm(resid))
if np.linalg.norm(resid) < 1e-8:
    print("  the constraint is satisfied to machine precision, so this really")
    print("  is a solution of the equations of motion and not an approximation.")
print()

print("--- 4 what the base weight buys, and what it costs ---")
# Report base motion RELATIVE to free fall. The absolute norm is about 9.81
# in every row because gravity is in it, and a column that reads 9.81 six
# times looks like a dead code path rather than a tradeoff.
qdd_free, _ = solve_qp(np.zeros(NV), wt_base=0.0)
print("  free fall baseline base accel norm: %.4f" % np.linalg.norm(qdd_free[:6]))
print("%12s %20s %16s" % ("base weight", "base accel vs free", "knee error"))
for wt in (0.0, 1.0, 10.0, 50.0, 500.0, 5000.0):
    qdd, tau = solve_qp(qdd_want, wt_base=wt)
    err = abs(qdd[VA[IDX["left_knee_joint"]]] - 2.0)
    dev = np.linalg.norm(qdd[:6] - qdd_free[:6])
    print("%12.1f %20.4f %16.4f" % (wt, dev, err))
print()
# Be careful about what this table supports. NEITHER column is monotone in the
# weight (base deviation goes 0.265, 0.203, 0.048, 0.076, 0.106, 0.147), so
# this is not a clean "turn the knob, trade one for the other" curve and I am
# not going to describe it as one. What it does show is the endpoints.
print("  at weight zero the knee gets exactly what it asked for (error 0.0000)")
print("  and the base deviates most. At weight 5000 the base is held tighter")
print("  and the knee error is 8.44 rad/s2, which is four times the request.")
print("  In between the relationship is NOT monotone, so this is not a dial")
print("  you tune by intuition.")
print()
print("  The structural fact underneath is simple: the base is unactuated, so")
print("  any knee acceleration MUST throw the body somewhere. The QP chooses")
print("  where, not whether, and that is all a priority scheme ever does.")
print()

print("--- 5 the same solve across the swing ---")
print("  driving the 5.5 swing angles through the QP at 5 points")
print("%8s %12s %12s %14s" % ("t", "knee deg", "knee tau", "base vs free"))
# t, hip deg, knee deg, from swing_ik.py
SW = [(0.000, -3.29, 28.92), (0.184, -15.40, 47.46), (0.368, -27.44, 54.88),
      (0.551, -32.06, 47.46), (0.735, -25.62, 28.92)]
peak = 0.0
for t, hdeg, kdeg in SW:
    # POSE the robot at this sample first. The mass matrix and the bias term
    # both depend on configuration, so solving them all at the spawn pose
    # returns the same torque five times and teaches nothing.
    d.qpos[:] = 0.0
    d.qpos[2] = 1.03
    d.qpos[3] = 1.0
    d.qpos[m.jnt_qposadr[m.actuator_trnid[IDX["left_hip_pitch_joint"]][0]]] = \
        np.radians(hdeg)
    d.qpos[m.jnt_qposadr[m.actuator_trnid[IDX["left_knee_joint"]][0]]] = \
        np.radians(kdeg)
    mujoco.mj_forward(m, d)
    mujoco.mj_fullM(m, d, M)
    h[:] = d.qfrc_bias
    dd = np.zeros(NV)
    dd[VA[IDX["left_knee_joint"]]] = 2.0
    dd[VA[IDX["left_hip_pitch_joint"]]] = -1.0
    qdd, tau = solve_qp(dd)
    tk = abs(tau[IDX["left_knee_joint"]])
    peak = max(peak, tk)
    print("%8.3f %12.2f %12.2f %14.4f"
          % (t, kdeg, tau[IDX["left_knee_joint"]],
             np.linalg.norm(qdd[:6] - qdd_free[:6])))
print()
LIM = float(m.jnt_actfrcrange[m.actuator_trnid[IDX["left_knee_joint"]][0]][1])
print("  peak knee torque demanded %.2f Nm against a %.0f Nm limit: %.0f%% used"
      % (peak, LIM, 100 * peak / LIM))
print()
print("that is the stack complete on paper: trajectory, foot placement, angles,")
print("torques. 5.7 runs it on the robot, and the robot falls over.")

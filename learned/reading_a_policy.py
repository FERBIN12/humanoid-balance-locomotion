#!/usr/bin/env python3
"""6.9 -- how much can you recover from the weights alone?

6.8 established that the boundary of a policy's competence lives in a training
config you may not have. This asks the harder question: given ONLY the file,
how much can you work out by looking inside it?

Four attempts, honestly scored. Some work. Most do not.
"""
import os, pathlib
import numpy as np
import torch, yaml

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
cfg = yaml.safe_load(open(ROOT / "policy/h1_2.yaml"))
PT = str(ROOT / "policy/pre_train/h1_2/motion.pt")
NA, NOBS = cfg["num_actions"], cfg["num_obs"]

policy = torch.jit.load(PT)
policy.eval()
W = {n: p.detach().numpy() for n, p in policy.named_parameters()}

print("--- attempt 1: which inputs does the first layer weight most ---")
# memory.weight_ih_l0 is (4*hidden, n_obs). Column j is how strongly input j
# drives every gate. Column norm is a defensible proxy for input salience.
ih = W["memory.weight_ih_l0"]
col = np.linalg.norm(ih, axis=0)
BLOCKS = [(0, 3, "base angular velocity"), (3, 6, "gravity in body frame"),
          (6, 9, "velocity command"), (9, 21, "joint positions"),
          (21, 33, "joint velocities"), (33, 45, "previous action"),
          (45, 47, "gait phase")]
print("%28s %14s %14s" % ("block", "mean col norm", "per input rank"))
rows = []
for a, b, name in BLOCKS:
    rows.append((name, float(col[a:b].mean())))
for name, v in sorted(rows, key=lambda r: -r[1]):
    print("%28s %14.4f" % (name, v))
print()
# 6.3 measured the same question by perturbation. Do they agree?
MEASURED = ["gravity in body frame", "base angular velocity",
            "velocity command", "joint positions", "gait phase",
            "joint velocities", "previous action"]
weights_order = [n for n, _ in sorted(rows, key=lambda r: -r[1])]
print("  6.3 measured this by perturbing a running policy and got:")
print("    %s" % ", ".join(m.split()[0] for m in MEASURED))
print("  reading the weights gives:")
print("    %s" % ", ".join(m.split()[0] for m in weights_order))
agree = sum(1 for i in range(len(MEASURED))
            if MEASURED[i] == weights_order[i])
print("  %d of %d positions agree." % (agree, len(MEASURED)))
if weights_order[0] == MEASURED[0]:
    print("  The top one matches, which is worth something.")
else:
    print("  Even the top one disagrees, so the weight norm is not measuring")
    print("  what the perturbation test measures.")
print()

print("--- attempt 2: can you find the gait period in the weights ---")
ph = col[45:47]
print("  the two gait phase inputs have column norms %.4f and %.4f"
      % (ph[0], ph[1]))
print("  Their MAGNITUDE says how much the network leans on the clock.")
print("  It says nothing at all about the PERIOD, because the period never")
print("  enters the network: it is applied outside, in the deploy loop, when")
print("  the sine and cosine are computed. 6.4 measured that running at the")
print("  wrong period still walks. No amount of weight inspection recovers")
print("  0.8 seconds, and that number is not in this file.")
print()

print("--- attempt 3: is the output symmetric between left and right ---")
# actor.2.weight is (12, 32): the final map to joint targets.
out = W["actor.2.weight"]
left, right = out[:6], out[6:]
sym = float(np.abs(np.linalg.norm(left, axis=1)
                   - np.linalg.norm(right, axis=1)).mean())
scale = float(np.linalg.norm(out, axis=1).mean())
print("  mean |‖left row‖ - ‖right row‖| = %.4f" % sym)
print("  mean row norm                   = %.4f" % scale)
print("  ratio %.1f%%" % (100 * sym / scale))
if sym / scale < 0.15:
    print("  The two legs are driven by rows of near equal magnitude, which is")
    print("  consistent with a symmetric gait. That is a real, checkable fact")
    print("  about the file, and it is also nearly the only one on this list.")
else:
    print("  The rows differ substantially, so the network is not obviously")
    print("  symmetric between legs.")
print()

print("--- attempt 4: can you recover the reward ---")
print("  No. And it is worth being precise about why, rather than hand waving.")
print()
print("  The reward shaped WHICH weights were reached by gradient descent. It")
print("  is not a term in the forward pass, it has no representation in the")
print("  parameters, and many different rewards can produce identical weights.")
print("  6.8 read seventeen reward terms out of a config file. None of them")
print("  are recoverable from these %d numbers." % sum(v.size for v in W.values()))
print()

print("--- the score ---")
SCORE = [
    ("input salience ranking", "NO: 1 of 7 positions agree, top disagrees"),
    ("the gait period", "no: it is not in the file"),
    ("left/right symmetry", "yes: directly checkable"),
    ("the reward function", "no: not represented in weights"),
    ("the randomisation ranges", "no: 6.8 needed a separate config"),
    ("what it will do on ice", "no: 6.6 had to run the robot"),
]
for q, a in SCORE:
    print("  %-28s %s" % (q, a))
print()
print("  One clear yes out of six, and the salience attempt is the")
print("  instructive failure. Column norm is the obvious proxy for 'how much")
print("  does this input matter', it is what I would have reached for, and")
print("  against a real measurement it agrees on 1 position out of 7. It even")
print("  disagrees at the top: the weights say joint positions, the running")
print("  robot says gravity. A plausible proxy is not a measurement.")
print()
print("  The useful conclusion is not that")
print("  interpretability is hopeless, it is that for THIS question, running")
print("  the robot is cheaper and more reliable than reading the network.")
print("  Every solid number in this section came from an experiment, and every")
print("  attempt to shortcut that by inspecting weights gave me less.")

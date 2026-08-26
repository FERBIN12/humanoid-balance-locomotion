#!/usr/bin/env python3
"""6.2 -- the policy file itself: what is actually in it.

We have been running motion.pt since section one without ever opening it. This
opens it. Not to reverse engineer the behaviour, which 6.9 attempts, but to
establish the plain facts: what shape it is, how big, what it takes in, what it
puts out, and what licence it is under.

Everything here is read off the file rather than quoted from a paper.
"""
import os, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
PT = ROOT / "policy/pre_train/h1_2/motion.pt"
LIC = ROOT / "policy/LICENSE.unitree_rl_gym"

print("--- 1 the file ---")
print("  path %s" % PT)
print("  size %.1f kB" % (PT.stat().st_size / 1024))
if LIC.exists():
    head = [l.strip() for l in LIC.read_text().splitlines() if l.strip()][:2]
    print("  licence file present: %s" % LIC.name)
    for l in head:
        print("    %s" % l[:70])
print()

policy = torch.jit.load(str(PT))
policy.eval()
print("--- 2 what kind of object it is ---")
print("  type: %s" % type(policy).__name__)
print("  a TorchScript module: the architecture is COMPILED IN, not described")
print("  by a separate config. You cannot edit the layer sizes without")
print("  retraining, and that is deliberate for deployment.")
print()

print("--- 3 the parameters, layer by layer ---")
total = 0
shapes = []
for name, p in policy.named_parameters():
    total += p.numel()
    shapes.append((name, tuple(p.shape), p.numel()))
    print("  %-28s %-16s %7d" % (name, str(tuple(p.shape)), p.numel()))
print("  %-28s %-16s %7d" % ("TOTAL", "", total))
print()
print("  %d parameters at 4 bytes each is %.1f kB of weights, against a"
      % (total, total * 4 / 1024))
print("  %.1f kB file. The rest is TorchScript's own bookkeeping."
      % (PT.stat().st_size / 1024))
print()

print("--- 4 THIS IS NOT A FEEDFORWARD NETWORK ---")
print("  The parameter names give it away: memory.weight_ih_l0 and")
print("  memory.weight_hh_l0 are the input-to-hidden and hidden-to-hidden")
print("  matrices of a recurrent layer. Reading the compiled forward pass")
print("  confirms it: the module holds hidden_state and cell_state, and")
print("  copies them back after every call. That is an LSTM.")
print()
print("  I described this policy as mapping the current state straight to")
print("  joint targets in 5.10, which implied a feedforward network and was")
print("  wrong. 5.10 and 5.11 have been corrected. Here is the demonstration")
print("  that settles it, feeding the SAME observation six times:")
obs = torch.zeros(1, 47)
with torch.no_grad():
    seq = [policy(obs).numpy().squeeze().copy() for _ in range(6)]
for i, a in enumerate(seq):
    print("    call %d -> action norm %.6f" % (i, np.linalg.norm(a)))
dev = max(float(np.abs(seq[i] - seq[0]).max()) for i in range(1, 6))
print("  max deviation from the first call: %.4f" % dev)
if dev > 1e-9:
    print("  A feedforward network returns identical output for identical")
    print("  input, always. This does not, because it remembers.")
print()

print("--- 5 what that means when you deploy it ---")
fresh = torch.jit.load(str(PT)); fresh.eval()
with torch.no_grad():
    f1 = fresh(obs).numpy().squeeze().copy()
    for _ in range(200):
        fresh(obs)
    warm = fresh(obs).numpy().squeeze().copy()
again = torch.jit.load(str(PT)); again.eval()
with torch.no_grad():
    f2 = again(obs).numpy().squeeze().copy()
print("  fresh load, first action norm     %.6f" % np.linalg.norm(f1))
print("  same object after 200 calls       %.6f" % np.linalg.norm(warm))
print("  a second fresh load, first action %.6f" % np.linalg.norm(f2))
print("  two fresh loads agree: %s" % np.allclose(f1, f2))
print()
print("  So the state starts zeroed and fresh loads are reproducible, but it")
print("  NEVER resets during a run. If you restart an episode without")
print("  reloading the module, the previous episode's state walks into the")
print("  new one. Every comparison in this project reloads, which is why the")
print("  numbers repeat, and that was luck rather than foresight.")
print()

print("--- 6 the shape of the computation ---")
lin = [(n, s) for n, s in [(a, b) for a, b, _ in shapes] if "weight" in n]
if lin:
    print("  reading the weight matrices in order:")
    dims = []
    for n, s in lin:
        if len(s) == 2:
            dims.append(s)
            print("    %-24s %d -> %d" % (n, s[1], s[0]))
    if dims:
        print()
        print("  so the network is %d in, %s hidden, %d out"
              % (dims[0][1], " -> ".join(str(d[0]) for d in dims[:-1]),
                 dims[-1][0]))
print()

print("--- 7 does it actually respond to its input ---")
# a policy that ignores its observations would still run and still produce
# numbers. Check that different states give different actions.
obs_zero = torch.zeros(1, 47)
with torch.no_grad():
    a0 = policy(obs_zero).numpy().squeeze()
    rng = np.random.default_rng(0)
    outs = []
    for _ in range(5):
        o = torch.from_numpy(rng.normal(0, 0.5, (1, 47)).astype(np.float32))
        outs.append(policy(o).numpy().squeeze())
outs = np.array(outs)
print("  zero observation  -> action norm %.4f" % np.linalg.norm(a0))
print("  5 random states   -> action norms %s"
      % np.round(np.linalg.norm(outs, axis=1), 3))
spread = float(np.abs(outs - outs.mean(0)).max())
print("  largest deviation between those actions: %.4f" % spread)
if spread > 0.05:
    print("  so the output genuinely depends on the input. That sounds trivial")
    print("  and is not: a policy loaded wrong, or fed an observation vector")
    print("  in the wrong order, still returns confident numbers.")
print()

print("--- 8 the interface, which is the part you have to get right ---")
print("  47 numbers in, 12 out, plus a hidden state you never see. The 12")
print("  are joint position TARGETS, not")
print("  torques: they are scaled and added to a default pose, then a PD")
print("  controller turns them into torque. The policy never commands a")
print("  torque directly, and it never sees one either.")
print()
print("  That means the policy is only half a controller. The other half is")
print("  the PD loop and its gains, which came from the manufacturer and which")
print("  section three already measured. Swap those gains and this policy's")
print("  behaviour changes without a single weight moving.")

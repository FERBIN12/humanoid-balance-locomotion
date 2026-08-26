#!/usr/bin/env python3
"""6.8 -- what the reward contained, and what that predicts.

I do not have Unitree's training config: the repository ships weights and a
deploy config, nothing else. What I do have is IsaacLab's H1 locomotion task,
which is the same problem posed the same way by a different group, with every
reward weight and randomisation range in readable Python.

So this reads a REAL config rather than describing rewards in the abstract, and
then checks whether it predicts the failures 6.6 measured.
"""
import os, pathlib, re

LAB = pathlib.Path(os.path.expanduser("~/IsaacLab"))
BASE = LAB / ("source/isaaclab_tasks/isaaclab_tasks/manager_based/"
              "locomotion/velocity/velocity_env_cfg.py")
H1 = LAB / ("source/isaaclab_tasks/isaaclab_tasks/manager_based/"
            "locomotion/velocity/config/h1/rough_env_cfg.py")

print("--- 1 where these numbers come from ---")
print("  Unitree ships motion.pt and a deploy config. No training config, no")
print("  reward function, no randomisation ranges. Those are not in the file")
print("  and 6.2 could not recover them from the weights.")
print("  So this reads IsaacLab's H1 task instead: same robot family, same")
print("  problem, different group, and every number visible.")
print("    %s" % BASE.name)
print("    %s" % H1.name)
print()


def rewards(path, cls):
    src = path.read_text()
    i = src.index("class " + cls)
    blk = src[i:i + 4000]
    out = []
    for m in re.finditer(r'^\s{4}(\w+)\s*=\s*(RewTerm\(|None)', blk, re.M):
        name = m.group(1)
        if m.group(2) == "None":
            out.append((name, None, "disabled"))
            continue
        seg = blk[m.start():m.start() + 500]
        w = re.search(r'weight=(-?[\d.e-]+)', seg)
        f = re.search(r'func=mdp\.(\w+)', seg)
        out.append((name, float(w.group(1)) if w else None,
                    f.group(1) if f else ""))
    return out


print("--- 2 the reward, term by term ---")
base = rewards(BASE, "RewardsCfg")
h1 = rewards(H1, "H1Rewards")
h1n = {n for n, _, _ in h1}
print("%28s %10s  %s" % ("term", "weight", "what it measures"))
for n, w, f in base:
    if n in h1n:
        continue
    print("%28s %10s  %s" % (n, ("%g" % w) if w is not None else "-", f))
print("  ... plus the H1 specific overrides:")
for n, w, f in h1:
    print("%28s %10s  %s" % (n, ("%g" % w) if w is not None else "-", f))
print()
pos = [(n, w) for n, w, _ in base + h1 if w and w > 0]
neg = [(n, w) for n, w, _ in base + h1 if w and w < 0]
print("  %d terms that PAY, %d that CHARGE." % (len(pos), len(neg)))
print("  paying:   %s" % ", ".join(n for n, _ in pos))
print("  charging: %s" % ", ".join(n for n, _ in neg))
print()
print("  Notice what is being paid for. Tracking the commanded velocity, and")
print("  keeping the feet in the air past a threshold. That is it. Nothing")
print("  rewards walking, or looking natural, or being efficient. Everything")
print("  else in this list is a penalty for a specific bad habit somebody")
print("  watched a robot develop.")
print()

print("--- 3 the domain randomisation, which 6.6 was really measuring ---")
src = BASE.read_text()
lines = src.splitlines()
i = next(k for k, l in enumerate(lines) if l.startswith("class EventsCfg"))
eb = "\n".join(lines[i:i + 90])
# Slice each event at the NEXT event's start, not a fixed 700 characters.
# A fixed window ran past the end of short terms and attributed the following
# term's interval and ranges to them, so reset_robot_joints appeared to have
# push_robot's 10 to 15 s interval.
starts = [m.start() for m in re.finditer(r'^\s{4}\w+\s*=\s*EventTerm\(',
                                         eb, re.M)] + [len(eb)]
for k, m in enumerate(re.finditer(r'^\s{4}(\w+)\s*=\s*EventTerm\(', eb, re.M)):
    seg = eb[m.start():starts[k + 1]]
    f = re.search(r'func=mdp\.(\w+)', seg)
    mode = re.search(r'mode="(\w+)"', seg)
    iv = re.search(r'interval_range_s=\(([\d.]+), ?([\d.]+)\)', seg)
    print("  %-26s %-30s mode=%s%s"
          % (m.group(1), f.group(1) if f else "",
             mode.group(1) if mode else "?",
             ("  every %s to %s s" % iv.groups()) if iv else ""))
    for r in re.findall(r'"(\w+_range)":\s*\(([-\d.]+),\s*([-\d.]+)\)', seg):
        span = float(r[2]) - float(r[1])
        print("       %-24s (%s, %s)%s"
              % (r[0], r[1], r[2], "   <- NO VARIATION" if span == 0 else ""))
print()

print("--- 4 does this predict what 6.6 measured ---")
print("%22s %26s %s" % ("6.6 result", "what training contained", "consistent"))
ROWS = [
    ("push: survives 450 N",
     "push_robot every 10-15 s", "yes"),
    ("friction: fails at 0.25",
     "friction fixed at 0.8, no range", "yes"),
    ("slope: fails at 3 deg",
     "no terrain tilt event at all", "yes"),
]
for a, b, c in ROWS:
    print("%22s %26s %s" % (a, b, c))
print()
print("  Three for three, and the friction one is the sharpest. I assumed")
print("  friction WAS randomised, because randomising it is standard advice.")
print("  The range is (0.8, 0.8): a single value with no variation at all.")
print("  So the policy never saw a slippery floor, and it fails on one.")
print()
print("  This is the answer to the question 6.6 left open. The boundary of")
print("  what a learned policy can handle is not in the weights and not in")
print("  the observation vector. It is in a config file somebody wrote before")
print("  training started, and if you do not have that file you are guessing.")

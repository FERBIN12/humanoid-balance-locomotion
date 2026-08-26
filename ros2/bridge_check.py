#!/usr/bin/env python3
"""9.4 -- what is actually crossing the topics.

The bringup works and the loop runs, so the next question is what the two
sides are actually saying to each other. Three things worth checking, and all
three have bitten me:

  1. the ORDER of joints in /joint_states against the controller's order
  2. what the command message actually contains
  3. which state interfaces exist and which are simulated fictions

The joint order one is the dangerous one. Both sides use plain arrays of
doubles with no names attached, so a mismatch is not an error, it is a robot
that moves the wrong limbs.
"""
import os
import xml.etree.ElementTree as ET
import pathlib

WS = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
URDF = WS / "src/cortex_humanoid_description/urdf/cortex_humanoid_torque.urdf"
CFG = WS / "src/cortex_humanoid_description/config/controllers.yaml"


def r2c_order():
    """The order ros2_control declares in the URDF."""
    r = ET.parse(URDF).getroot()
    out = []
    for b in r.findall("ros2_control"):
        for j in b.findall("joint"):
            out.append(j.get("name"))
    return out


def cfg_order():
    """The order the effort controller lists in its yaml."""
    import re
    s = CFG.read_text()
    m = re.search(r"effort_controller:\s*\n\s*ros__parameters:\s*\n\s*joints:\s*\n((?:\s*-\s*\S+\n)+)", s)
    if not m:
        return []
    return re.findall(r"-\s*(\S+)", m.group(1))


def interfaces():
    r = ET.parse(URDF).getroot()
    cmd, st = set(), set()
    for b in r.findall("ros2_control"):
        for j in b.findall("joint"):
            for c in j.findall("command_interface"):
                cmd.add(c.get("name"))
            for c in j.findall("state_interface"):
                st.add(c.get("name"))
    return sorted(cmd), sorted(st)


if __name__ == "__main__":
    a, b = r2c_order(), cfg_order()
    print("--- joint ORDER, the one that fails silently ---")
    print(f"  the URDF's ros2_control block lists {len(a)} joints")
    print(f"  controllers.yaml lists {len(b)}")
    if b:
        same = a == b
        print(f"  identical order: {same}")
        if not same:
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    print(f"    first divergence at index {i}: "
                          f"{x} vs {y}")
                    break
    print()
    print("  Both sides send a plain array of doubles with no names in it.")
    print("  A mismatch here is not an error and not a warning. It is a")
    print("  robot that moves the wrong limbs, confidently, at 400 Hz.")
    print()

    cmd, st = interfaces()
    print("--- interfaces ---")
    print(f"  command: {cmd}")
    print(f"  state:   {st}")
    print()
    print("  Note that 'effort' appears in BOTH. On this simulated robot the")
    print("  effort state interface reports back the effort that was")
    print("  commanded, which is not a measurement of anything. On hardware")
    print("  it would come from a current sensor or a strain gauge and would")
    print("  differ from the command, which is the entire point of having it.")
    print("  A controller that closes a loop on simulated effort state is")
    print("  closing a loop on its own output.")

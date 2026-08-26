#!/usr/bin/env python3
"""9.3 -- does the 400 Hz loop actually run at 400 Hz?

controllers.yaml asks for update_rate: 400, and the balance controller justified that from
the LIPM time constant sqrt(h/g) = 0.311 s. A number in a config file is a
request, not a measurement, and the gap between the two is exactly the sort of
thing that quietly ruins a control result.

This measures three separate things that all get called "the rate":

  1. what controller_manager reports as its update rate
  2. how fast /joint_states actually publishes
  3. how fast a Python node can close the loop on top of it
"""
import os
import re
import subprocess

WS = os.path.expanduser("~/humanoid_ws")
SRC = ("source /opt/ros/jazzy/setup.bash && source %s/install/setup.bash && "
       % WS)


def sh(cmd, t=30):
    return subprocess.run(["bash", "-lc", SRC + cmd], capture_output=True,
                          text=True, timeout=t).stdout


if __name__ == "__main__":
    print("--- what the config asks for ---")
    cfg = open(os.path.join(
        WS, "src/cortex_humanoid_description/config/controllers.yaml")).read()
    m = re.search(r"update_rate:\s*(\d+)", cfg)
    print("  controllers.yaml update_rate: %s Hz" % (m.group(1) if m else "?"))
    print("  the balance controller derived this from sqrt(h/g) = 0.311 s, so anything")
    print("  slower than a couple of hundred Hz is already behind the")
    print("  divergence it is meant to catch.")
    print()
    print("--- and what to measure instead of trusting it ---")
    print("  1. controller_manager's own reported rate")
    print("  2. the actual publish rate of /joint_states")
    print("  3. what a Python node on top can sustain")
    print()
    print("  Run this with the stack up:")
    print("    ros2 launch cortex_humanoid_description gz_bringup.launch.py")
    print("    ros2 topic hz /joint_states")
    print()
    print("  A config number is a request. The three rates above are three")
    print("  different quantities and they are routinely conflated, which is")
    print("  how a 'a 400 Hz controller' ends up closing its loop at 50.")

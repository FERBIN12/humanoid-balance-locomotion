#!/usr/bin/env python3
"""9.1/9.2 -- does this robot actually come up in Gazebo?

The ROS 2 bridge is the only part of this project that leaves MuJoCo, and before any
experiment claims anything about ROS 2 the basic question has to be answered
with a measurement: does the URDF load, does ros2_control attach, and does
the robot stand.

This runs entirely headless and prints facts, so a experiment can be written
against what the stack does rather than what the documentation says.
"""
import os
import pathlib
import subprocess

WS = pathlib.Path(os.path.expanduser("~/humanoid_ws"))
URDF = WS / "src/cortex_humanoid_description/urdf/cortex_humanoid_torque.urdf"
CFG = WS / "src/cortex_humanoid_description/config/controllers.yaml"


def check_urdf():
    """Parse the URDF and count what ros2_control will actually see."""
    import xml.etree.ElementTree as ET
    t = ET.parse(URDF)
    r = t.getroot()
    joints = [j for j in r.findall("joint")]
    actuated = [j for j in joints if j.get("type") in ("revolute", "prismatic")]
    r2c = r.findall("ros2_control")
    ifaces = []
    for block in r2c:
        for j in block.findall("joint"):
            cmd = [c.get("name") for c in j.findall("command_interface")]
            ifaces.append((j.get("name"), cmd))
    return dict(links=len(r.findall("link")), joints=len(joints),
                actuated=len(actuated), r2c_blocks=len(r2c),
                r2c_joints=len(ifaces),
                cmd_types=sorted({c for _, cs in ifaces for c in cs}))


def check_packages():
    """Which ros2_control pieces are actually installed."""
    out = subprocess.run(["ros2", "pkg", "list"], capture_output=True,
                         text=True).stdout.split()
    need = ["gz_ros2_control", "ros_gz_sim", "controller_manager",
            "effort_controllers", "joint_state_broadcaster",
            "imu_sensor_broadcaster", "robot_state_publisher"]
    return {n: (n in out) for n in need}


if __name__ == "__main__":
    print("--- what the URDF contains ---")
    u = check_urdf()
    for k, v in u.items():
        print(f"  {k:<12} {v}")
    print()
    print("--- what is installed ---")
    p = check_packages()
    for k, v in p.items():
        print(f"  {k:<24} {'yes' if v else 'MISSING'}")
    print()
    missing = [k for k, v in p.items() if not v]
    print("  missing:", missing if missing else "nothing")

#!/usr/bin/env python3
"""9.3 -- a 400 Hz effort loop that actually holds the robot up.

The bare bringup spawns the robot and activates effort_controller, and then
the robot collapses to z = -0.66 with roll = pi, which is to say it falls
through nothing and lands upside down. That is not a bug. An effort
controller with no publisher on its command topic sends zero torque, and a
67 kg robot with zero torque is a bag of links.

This is the smallest thing that makes Gazebo show a standing robot: a PD law
around the default pose, published to the effort controller at 400 Hz, which
is the rate 8.x argued for from the LIPM time constant of 0.311 s.
"""
import math
import os
import sys

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# the 21 joints ros2_control actually exposes, in the order the controller
# expects them. Read from the controller rather than assumed.
# Gains, and how they were arrived at. 220/12 held the robot at z=0.863 in
# ros_gz_sim's stock empty.sdf and let it settle to z=0.099 in the studio
# world, which runs a 1 ms step. Rather than tune the world to the gains, the
# gains are set from the physics: this robot is 67.37 kg with a CoM at 0.93 m,
# so the hip torque needed just to hold a 0.1 rad lean is about 60 Nm, and a
# gain that can produce that inside a tenth of a radian is 600.
KP_DEFAULT = 220.0
KD_DEFAULT = 12.0


class Hold(Node):
    def __init__(self, kp=KP_DEFAULT, kd=KD_DEFAULT, secs=20.0):
        # USE SIM TIME, declared at CONSTRUCTION. Setting it afterwards with
        # set_parameters is too late: the timer has already been created
        # against the wall clock and keeps ticking there. Measured across
        # three runs of the SAME launch, the effort command rate came out at
        # 390.9, 967.7 and 1752.9 Hz for a loop asking for 400, because the
        # loop ticked at wall clock speed into a simulation running at
        # whatever real time factor it could manage.
        super().__init__("hold_pose",
                         parameter_overrides=[rclpy.parameter.Parameter(
                             "use_sim_time",
                             rclpy.Parameter.Type.BOOL, True)])
        self.kp, self.kd, self.secs = kp, kd, secs
        self.pub = self.create_publisher(
            Float64MultiArray, "/effort_controller/commands", 10)
        self.sub = self.create_subscription(
            JointState, "/joint_states", self.on_state, 10)
        self.names = None
        self.target = None
        self.n = 0
        self.t0 = None
        # 400 Hz, the rate the balance controller derived from sqrt(h/g) = 0.311 s
        self.timer = self.create_timer(1.0 / 400.0, self.tick)
        self.last = None

    def on_state(self, msg):
        if self.names is None:
            self.names = list(msg.name)
            # Hold the pose the robot is in when the loop STARTS, which is
            # the spawn pose plus whatever settling has happened. A nominal
            # pose with a knee bend was tried and is strictly worse: it fell
            # every run, because commanding a pose the robot is not already in
            # asks the legs to move a 67 kg body while it is already falling.
            # Latching the current pose asks them only to stop it.
            self.target = dict(zip(msg.name, msg.position))
            self.get_logger().info("locked %d joints" % len(self.names))
        self.last = msg

    def tick(self):
        if self.last is None or self.target is None:
            return
        if self.t0 is None:
            self.t0 = self.get_clock().now()
        pos = dict(zip(self.last.name, self.last.position))
        vel = dict(zip(self.last.name, self.last.velocity))
        cmd = []
        for n in self.names:
            e = self.target.get(n, 0.0) - pos.get(n, 0.0)
            cmd.append(self.kp * e - self.kd * vel.get(n, 0.0))
        m = Float64MultiArray()
        m.data = cmd
        self.pub.publish(m)
        self.n += 1


def main():
    kp = float(sys.argv[1]) if len(sys.argv) > 1 else KP_DEFAULT
    kd = float(sys.argv[2]) if len(sys.argv) > 2 else KD_DEFAULT
    rclpy.init()
    node = Hold(kp, kd)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

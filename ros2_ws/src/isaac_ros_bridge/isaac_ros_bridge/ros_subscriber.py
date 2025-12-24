#!/usr/bin/env python3
"""
ROS2 to Isaac Sim Subscriber Node.

This node subscribes to ROS2 topics and forwards commands to Isaac Sim:
- /cmd_vel: Velocity commands (Twist messages)

Usage:
    ros2 run isaac_ros_bridge ros_subscriber
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import numpy as np
from typing import Callable


class RosSubscriber(Node):
    """ROS2 node that subscribes to commands for Isaac Sim."""

    def __init__(self):
        super().__init__("ros_subscriber")

        # Declare parameters
        self.declare_parameter("max_linear_velocity", 2.0)
        self.declare_parameter("max_angular_velocity", 1.5)
        self.declare_parameter("cmd_timeout", 0.5)

        # Get parameters
        self.max_linear = self.get_parameter("max_linear_velocity").value
        self.max_angular = self.get_parameter("max_angular_velocity").value
        self.cmd_timeout = self.get_parameter("cmd_timeout").value

        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )

        # State
        self._last_cmd_time = self.get_clock().now()
        self._linear_velocity = 0.0
        self._angular_velocity = 0.0

        # Callback for Isaac Sim (set externally)
        self._isaac_callback: Callable[[float, float], None] | None = None

        # Timer to check for command timeout
        self.timer = self.create_timer(0.1, self.timeout_callback)

        self.get_logger().info("ROS Subscriber initialized")

    def set_isaac_callback(self, callback: Callable[[float, float], None]) -> None:
        """Set callback function to send commands to Isaac Sim.

        Args:
            callback: Function that takes (linear_velocity, angular_velocity)
        """
        self._isaac_callback = callback

    def cmd_vel_callback(self, msg: Twist) -> None:
        """Handle velocity command messages."""
        # Clamp velocities to limits
        self._linear_velocity = np.clip(
            msg.linear.x, -self.max_linear, self.max_linear
        )
        self._angular_velocity = np.clip(
            msg.angular.z, -self.max_angular, self.max_angular
        )
        self._last_cmd_time = self.get_clock().now()

        # Forward to Isaac Sim if callback is set
        if self._isaac_callback is not None:
            self._isaac_callback(self._linear_velocity, self._angular_velocity)

        self.get_logger().debug(
            f"Received cmd_vel: linear={self._linear_velocity:.2f}, "
            f"angular={self._angular_velocity:.2f}"
        )

    def timeout_callback(self) -> None:
        """Check for command timeout and stop robot if no commands received."""
        elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9

        if elapsed > self.cmd_timeout:
            if self._linear_velocity != 0.0 or self._angular_velocity != 0.0:
                self._linear_velocity = 0.0
                self._angular_velocity = 0.0

                if self._isaac_callback is not None:
                    self._isaac_callback(0.0, 0.0)

                self.get_logger().info("Command timeout - stopping robot")

    @property
    def linear_velocity(self) -> float:
        """Get current linear velocity command."""
        return self._linear_velocity

    @property
    def angular_velocity(self) -> float:
        """Get current angular velocity command."""
        return self._angular_velocity


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = RosSubscriber()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

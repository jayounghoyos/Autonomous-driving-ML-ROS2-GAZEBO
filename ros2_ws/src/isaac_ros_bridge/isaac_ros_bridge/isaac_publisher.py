#!/usr/bin/env python3
"""
Isaac Sim to ROS2 Publisher Node.

This node publishes robot state data from Isaac Sim to ROS2 topics:
- /odom: Robot odometry (position, orientation, velocities)
- /camera/image_raw: RGB camera images (if enabled)
- /scan: LiDAR scan data (if enabled)

Usage:
    ros2 run isaac_ros_bridge isaac_publisher
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import TransformBroadcaster

import numpy as np


class IsaacPublisher(Node):
    """ROS2 node that publishes data from Isaac Sim."""

    def __init__(self):
        super().__init__("isaac_publisher")

        # Declare parameters
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)

        # Get parameters
        self.publish_rate = self.get_parameter("publish_rate").value
        self.frame_id = self.get_parameter("frame_id").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.publish_tf = self.get_parameter("publish_tf").value

        # Publishers
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)

        # TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)

        # Timer for publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_callback)

        # State (to be updated by Isaac Sim)
        self._position = np.zeros(3)
        self._orientation = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        self._linear_velocity = np.zeros(3)
        self._angular_velocity = np.zeros(3)
        self._image = None
        self._scan = None

        self.get_logger().info("Isaac Publisher initialized")

    def set_robot_state(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        linear_velocity: np.ndarray | None = None,
        angular_velocity: np.ndarray | None = None,
    ) -> None:
        """Update robot state from Isaac Sim.

        Args:
            position: [x, y, z] position in meters
            orientation: [w, x, y, z] quaternion
            linear_velocity: [vx, vy, vz] in m/s
            angular_velocity: [wx, wy, wz] in rad/s
        """
        self._position = position
        self._orientation = orientation
        if linear_velocity is not None:
            self._linear_velocity = linear_velocity
        if angular_velocity is not None:
            self._angular_velocity = angular_velocity

    def set_camera_image(self, image: np.ndarray) -> None:
        """Update camera image from Isaac Sim.

        Args:
            image: RGB image as numpy array (H, W, 3)
        """
        self._image = image

    def set_lidar_scan(self, ranges: np.ndarray, angle_min: float, angle_max: float) -> None:
        """Update LiDAR scan from Isaac Sim.

        Args:
            ranges: Range measurements in meters
            angle_min: Start angle in radians
            angle_max: End angle in radians
        """
        self._scan = (ranges, angle_min, angle_max)

    def publish_callback(self) -> None:
        """Timer callback to publish data."""
        now = self.get_clock().now().to_msg()

        # Publish odometry
        self._publish_odom(now)

        # Publish TF
        if self.publish_tf:
            self._publish_tf(now)

        # Publish camera image
        if self._image is not None:
            self._publish_image(now)

        # Publish LiDAR scan
        if self._scan is not None:
            self._publish_scan(now)

    def _publish_odom(self, timestamp) -> None:
        """Publish odometry message."""
        msg = Odometry()
        msg.header.stamp = timestamp
        msg.header.frame_id = self.frame_id
        msg.child_frame_id = self.child_frame_id

        # Position
        msg.pose.pose.position.x = float(self._position[0])
        msg.pose.pose.position.y = float(self._position[1])
        msg.pose.pose.position.z = float(self._position[2])

        # Orientation (quaternion)
        msg.pose.pose.orientation.w = float(self._orientation[0])
        msg.pose.pose.orientation.x = float(self._orientation[1])
        msg.pose.pose.orientation.y = float(self._orientation[2])
        msg.pose.pose.orientation.z = float(self._orientation[3])

        # Velocity
        msg.twist.twist.linear.x = float(self._linear_velocity[0])
        msg.twist.twist.linear.y = float(self._linear_velocity[1])
        msg.twist.twist.linear.z = float(self._linear_velocity[2])
        msg.twist.twist.angular.x = float(self._angular_velocity[0])
        msg.twist.twist.angular.y = float(self._angular_velocity[1])
        msg.twist.twist.angular.z = float(self._angular_velocity[2])

        self.odom_pub.publish(msg)

    def _publish_tf(self, timestamp) -> None:
        """Publish TF transform."""
        t = TransformStamped()
        t.header.stamp = timestamp
        t.header.frame_id = self.frame_id
        t.child_frame_id = self.child_frame_id

        t.transform.translation.x = float(self._position[0])
        t.transform.translation.y = float(self._position[1])
        t.transform.translation.z = float(self._position[2])

        t.transform.rotation.w = float(self._orientation[0])
        t.transform.rotation.x = float(self._orientation[1])
        t.transform.rotation.y = float(self._orientation[2])
        t.transform.rotation.z = float(self._orientation[3])

        self.tf_broadcaster.sendTransform(t)

    def _publish_image(self, timestamp) -> None:
        """Publish camera image."""
        if self._image is None:
            return

        msg = Image()
        msg.header.stamp = timestamp
        msg.header.frame_id = "camera_link"
        msg.height = self._image.shape[0]
        msg.width = self._image.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = self._image.shape[1] * 3
        msg.data = self._image.tobytes()

        self.image_pub.publish(msg)

    def _publish_scan(self, timestamp) -> None:
        """Publish LiDAR scan."""
        if self._scan is None:
            return

        ranges, angle_min, angle_max = self._scan

        msg = LaserScan()
        msg.header.stamp = timestamp
        msg.header.frame_id = "lidar_link"
        msg.angle_min = angle_min
        msg.angle_max = angle_max
        msg.angle_increment = (angle_max - angle_min) / len(ranges)
        msg.time_increment = 0.0
        msg.scan_time = 1.0 / self.publish_rate
        msg.range_min = 0.1
        msg.range_max = 20.0
        msg.ranges = ranges.tolist()

        self.scan_pub.publish(msg)


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    node = IsaacPublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

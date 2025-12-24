#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math


class PIDController(Node):
    """
    Simple PID controller for following waypoints.
    Subscribes to /odom and target waypoint, publishes /cmd_vel
    """

    def __init__(self):
        super().__init__('pid_controller')

        # PID gains
        self.kp_linear = 1.0
        self.ki_linear = 0.0
        self.kd_linear = 0.1

        self.kp_angular = 2.0
        self.ki_angular = 0.0
        self.kd_angular = 0.1

        # State variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0

        self.target_x = 5.0  # Default target
        self.target_y = 5.0

        self.prev_error_linear = 0.0
        self.prev_error_angular = 0.0
        self.integral_linear = 0.0
        self.integral_angular = 0.0

        # Publishers and subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop timer (50 Hz)
        self.timer = self.create_timer(0.02, self.control_loop)

        self.get_logger().info('PID Controller Node Started')
        self.get_logger().info(f'Target: ({self.target_x}, {self.target_y})')

    def odom_callback(self, msg):
        """Update current position and orientation"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y

        # Convert quaternion to yaw
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y * orientation.y + orientation.z * orientation.z)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

    def control_loop(self):
        """PID control loop"""
        # Calculate distance and angle to target
        dx = self.target_x - self.current_x
        dy = self.target_y - self.current_y
        distance = math.sqrt(dx**2 + dy**2)

        target_angle = math.atan2(dy, dx)
        angle_error = self.normalize_angle(target_angle - self.current_yaw)

        # PID for linear velocity
        error_linear = distance
        self.integral_linear += error_linear * 0.02
        derivative_linear = (error_linear - self.prev_error_linear) / 0.02

        linear_vel = (self.kp_linear * error_linear +
                     self.ki_linear * self.integral_linear +
                     self.kd_linear * derivative_linear)

        # PID for angular velocity
        self.integral_angular += angle_error * 0.02
        derivative_angular = (angle_error - self.prev_error_angular) / 0.02

        angular_vel = (self.kp_angular * angle_error +
                      self.ki_angular * self.integral_angular +
                      self.kd_angular * derivative_angular)

        # Publish command
        twist = Twist()

        if distance < 0.5:
            # Reached target
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info('Target reached!', throttle_duration_sec=1.0)
        else:
            twist.linear.x = min(linear_vel, 2.0)  # Max speed 2 m/s
            twist.angular.z = max(min(angular_vel, 1.0), -1.0)  # Max turn rate ±1 rad/s

        self.cmd_vel_pub.publish(twist)

        # Update previous errors
        self.prev_error_linear = error_linear
        self.prev_error_angular = angle_error

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


def main(args=None):
    rclpy.init(args=args)
    node = PIDController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

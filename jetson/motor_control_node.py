#!/usr/bin/env python3
"""
ROS2 Motor Control Node for Shadow Robot.

Subscribes to /cmd_vel and controls L298N motor driver.

Wiring:
  IN1 → Pin 11 - Left motor direction 1
  IN2 → Pin 12 - Left motor direction 2
  IN3 → Pin 13 - Right motor direction 1
  IN4 → Pin 15 - Right motor direction 2
  ENA/ENB → Jumpered (full speed)

Run on Jetson Nano:
  ros2 run shadow_robot motor_control_node

Or directly:
  python3 motor_control_node.py
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    import Jetson.GPIO as GPIO
    JETSON_AVAILABLE = True
except ImportError:
    JETSON_AVAILABLE = False
    print("WARNING: Jetson.GPIO not available. Running in simulation mode.")


class MotorControlNode(Node):
    """ROS2 node for controlling differential drive motors via L298N."""

    # Pin definitions (BOARD numbering)
    IN1 = 11  # Left motor direction 1
    IN2 = 12  # Left motor direction 2
    IN3 = 13  # Right motor direction 1
    IN4 = 15  # Right motor direction 2

    # Thresholds
    DEADZONE = 0.1  # Ignore commands smaller than this

    def __init__(self):
        super().__init__('motor_control_node')

        # Initialize GPIO
        if JETSON_AVAILABLE:
            GPIO.setmode(GPIO.BOARD)
            GPIO.setwarnings(False)
            GPIO.setup([self.IN1, self.IN2, self.IN3, self.IN4], GPIO.OUT)
            self.stop_motors()
            self.get_logger().info("GPIO initialized")
        else:
            self.get_logger().warn("GPIO not available - simulation mode")

        # Subscribe to velocity commands
        self.subscription = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )

        # Safety timer - stop if no commands received
        self.last_cmd_time = self.get_clock().now()
        self.safety_timer = self.create_timer(0.5, self.safety_check)

        self.get_logger().info("Motor Control Node started")
        self.get_logger().info("  Subscribing to: /cmd_vel")
        self.get_logger().info("  Pins: IN1=%d, IN2=%d, IN3=%d, IN4=%d" %
                               (self.IN1, self.IN2, self.IN3, self.IN4))

    def cmd_vel_callback(self, msg: Twist):
        """Handle incoming velocity commands."""
        self.last_cmd_time = self.get_clock().now()

        linear = msg.linear.x   # Forward/backward (-1 to 1)
        angular = msg.angular.z  # Turn left/right (-1 to 1)

        # Differential drive mixing
        # Left wheel = linear - angular
        # Right wheel = linear + angular
        left_speed = linear - angular
        right_speed = linear + angular

        # Clamp to [-1, 1]
        left_speed = max(-1.0, min(1.0, left_speed))
        right_speed = max(-1.0, min(1.0, right_speed))

        # Apply to motors
        self.set_motor_direction('left', left_speed)
        self.set_motor_direction('right', right_speed)

        self.get_logger().debug(
            f"cmd_vel: linear={linear:.2f}, angular={angular:.2f} -> "
            f"left={left_speed:.2f}, right={right_speed:.2f}"
        )

    def set_motor_direction(self, side: str, speed: float):
        """Set motor direction based on speed value."""
        if not JETSON_AVAILABLE:
            return

        if side == 'left':
            in1, in2 = self.IN1, self.IN2
        else:
            in1, in2 = self.IN3, self.IN4

        if speed > self.DEADZONE:
            # Forward
            GPIO.output(in1, GPIO.HIGH)
            GPIO.output(in2, GPIO.LOW)
        elif speed < -self.DEADZONE:
            # Backward
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.HIGH)
        else:
            # Stop (within deadzone)
            GPIO.output(in1, GPIO.LOW)
            GPIO.output(in2, GPIO.LOW)

    def stop_motors(self):
        """Stop all motors."""
        if JETSON_AVAILABLE:
            GPIO.output(self.IN1, GPIO.LOW)
            GPIO.output(self.IN2, GPIO.LOW)
            GPIO.output(self.IN3, GPIO.LOW)
            GPIO.output(self.IN4, GPIO.LOW)

    def safety_check(self):
        """Stop motors if no commands received recently."""
        now = self.get_clock().now()
        elapsed = (now - self.last_cmd_time).nanoseconds / 1e9

        if elapsed > 1.0:  # 1 second timeout
            self.stop_motors()

    def destroy_node(self):
        """Clean up on shutdown."""
        self.get_logger().info("Shutting down motor control...")
        self.stop_motors()
        if JETSON_AVAILABLE:
            GPIO.cleanup()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    node = MotorControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

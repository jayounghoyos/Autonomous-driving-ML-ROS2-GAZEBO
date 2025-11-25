#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import termios
import tty


class AckermannTeleop(Node):
    """
    Keyboard teleoperation for Ackermann steering vehicle.
    Allows simultaneous forward/backward and steering.
    """

    def __init__(self):
        super().__init__('ackermann_teleop')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(0.1, self.publish_cmd)

        self.linear_speed = 0.0
        self.angular_speed = 0.0

        self.max_linear = 3.0  # m/s
        self.max_angular = 0.6  # rad (steering angle)

        self.speed_increment = 0.5
        self.steering_increment = 0.2

        self.get_logger().info('Ackermann Keyboard Teleoperation Started')
        self.get_logger().info('Controls:')
        self.get_logger().info('  W/S - Increase/Decrease Speed')
        self.get_logger().info('  A/D - Steer Left/Right')
        self.get_logger().info('  X   - Emergency Stop')
        self.get_logger().info('  Q   - Quit')
        self.get_logger().info('')
        self.get_logger().info('Current - Speed: 0.0, Steering: 0.0')

    def get_key(self):
        """Non-blocking key read"""
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def publish_cmd(self):
        """Publish current velocity command"""
        twist = Twist()
        twist.linear.x = self.linear_speed
        twist.angular.z = self.angular_speed
        self.publisher.publish(twist)

    def run(self):
        """Main teleoperation loop"""
        self.settings = termios.tcgetattr(sys.stdin)

        try:
            while rclpy.ok():
                key = self.get_key()

                if key == 'w' or key == 'W':
                    self.linear_speed = min(self.linear_speed + self.speed_increment, self.max_linear)
                    self.get_logger().info(f'Speed: {self.linear_speed:.1f} m/s, Steering: {self.angular_speed:.2f} rad')

                elif key == 's' or key == 'S':
                    self.linear_speed = max(self.linear_speed - self.speed_increment, -self.max_linear)
                    self.get_logger().info(f'Speed: {self.linear_speed:.1f} m/s, Steering: {self.angular_speed:.2f} rad')

                elif key == 'a' or key == 'A':
                    self.angular_speed = max(self.angular_speed - self.steering_increment, -self.max_angular)
                    self.get_logger().info(f'Speed: {self.linear_speed:.1f} m/s, Steering: {self.angular_speed:.2f} rad (LEFT)')

                elif key == 'd' or key == 'D':
                    self.angular_speed = min(self.angular_speed + self.steering_increment, self.max_angular)
                    self.get_logger().info(f'Speed: {self.linear_speed:.1f} m/s, Steering: {self.angular_speed:.2f} rad (RIGHT)')

                elif key == 'x' or key == 'X':
                    self.linear_speed = 0.0
                    self.angular_speed = 0.0
                    self.get_logger().info('EMERGENCY STOP')

                elif key == ' ':
                    # Gradually reduce speed
                    if abs(self.linear_speed) > 0.1:
                        self.linear_speed *= 0.8
                    else:
                        self.linear_speed = 0.0
                    # Return steering to center
                    if abs(self.angular_speed) > 0.05:
                        self.angular_speed *= 0.7
                    else:
                        self.angular_speed = 0.0
                    self.get_logger().info(f'Coasting... Speed: {self.linear_speed:.1f}, Steering: {self.angular_speed:.2f}')

                elif key == 'q' or key == 'Q':
                    self.get_logger().info('Quitting...')
                    break

                rclpy.spin_once(self, timeout_sec=0)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

        finally:
            # Stop the vehicle
            twist = Twist()
            self.publisher.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannTeleop()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

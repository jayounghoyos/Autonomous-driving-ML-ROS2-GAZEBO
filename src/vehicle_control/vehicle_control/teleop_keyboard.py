#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty


class TeleopKeyboard(Node):

    def __init__(self):
        super().__init__('teleop_keyboard')

        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.linear_speed = 2.0  # m/s
        self.angular_speed = 1.0  # rad/s

        self.get_logger().info('Keyboard Teleoperation Node Started')
        self.get_logger().info('Use WASD keys to control the vehicle:')
        self.get_logger().info('  W - Forward')
        self.get_logger().info('  S - Backward')
        self.get_logger().info('  A - Turn Left')
        self.get_logger().info('  D - Turn Right')
        self.get_logger().info('  Space - Stop')
        self.get_logger().info('  Q - Quit')

    def get_key(self):
        """Get a single keypress from terminal"""
        tty.setraw(sys.stdin.fileno())
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        """Main teleoperation loop"""
        self.settings = termios.tcgetattr(sys.stdin)

        try:
            while True:
                key = self.get_key()

                twist = Twist()

                if key == 'w' or key == 'W':
                    twist.linear.x = self.linear_speed
                    self.get_logger().info('Forward')

                elif key == 's' or key == 'S':
                    twist.linear.x = -self.linear_speed
                    self.get_logger().info('Backward')

                elif key == 'a' or key == 'A':
                    twist.angular.z = self.angular_speed
                    self.get_logger().info('Turn Left')

                elif key == 'd' or key == 'D':
                    twist.angular.z = -self.angular_speed
                    self.get_logger().info('Turn Right')

                elif key == ' ':
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.get_logger().info('Stop')

                elif key == 'q' or key == 'Q':
                    self.get_logger().info('Quitting...')
                    break

                else:
                    continue

                self.publisher.publish(twist)

        except Exception as e:
            self.get_logger().error(f'Error: {e}')

        finally:
            # Stop the vehicle
            twist = Twist()
            self.publisher.publish(twist)
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)


def main(args=None):
    rclpy.init(args=args)

    node = TeleopKeyboard()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import select
import tty
import termios

class SimpleDriver(Node):
    def __init__(self):
        super().__init__('simple_driver')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.speed = 0.0
        self.turn = 0.0
        self.get_logger().info('Simple WASD Driver Started')
        self.get_logger().info('Controls: W (Accel), S (Brake/Rev), A (Left), D (Right), Space (Stop)')
        self.get_logger().info('Press CTRL+C to exit')

    def publish_command(self):
        msg = Twist()
        msg.linear.x = self.speed
        msg.angular.z = self.turn
        self.publisher_.publish(msg)

def get_key():
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
    if rlist:
        key = sys.stdin.read(1)
    else:
        key = ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key

settings = termios.tcgetattr(sys.stdin)

def main(args=None):
    rclpy.init(args=args)
    node = SimpleDriver()
    
    speed_step = 0.5
    turn_step = 0.5
    max_speed = 5.0
    max_turn = 1.0

    try:
        while rclpy.ok():
            key = get_key()
            if key == 'w':
                node.speed = min(node.speed + speed_step, max_speed)
                print(f"Speed: {node.speed:.1f}", end='\r')
            elif key == 's':
                node.speed = max(node.speed - speed_step, -max_speed)
                print(f"Speed: {node.speed:.1f}", end='\r')
            elif key == 'a':
                node.turn = min(node.turn + turn_step, max_turn)
                print(f"Turn: {node.turn:.1f}", end='\r')
            elif key == 'd':
                node.turn = max(node.turn - turn_step, -max_turn)
                print(f"Turn: {node.turn:.1f}", end='\r')
            elif key == ' ':
                node.speed = 0.0
                node.turn = 0.0
                print("STOP                    ", end='\r')
            elif key == '\x03': # Ctrl+C
                break
            
            node.publish_command()
            #rclpy.spin_once(node, timeout_sec=0.1) 

    except Exception as e:
        print(e)
    finally:
        node.publish_command() # Stop on exit
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

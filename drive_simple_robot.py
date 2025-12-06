#!/usr/bin/env python3
"""
Manual control for simple_robot using joint forces
Press W/S to move forward/backward, A/D to turn
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
import sys, select, tty, termios

class SimpleRobotDriver(Node):
    def __init__(self):
        super().__init__('simple_robot_driver')
        
        # Publishers for joint forces
        self.left_pub = self.create_publisher(
            Float64, '/model/simple_robot/joint/left_wheel_joint/cmd_force', 10)
        self.right_pub = self.create_publisher(
            Float64, '/model/simple_robot/joint/right_wheel_joint/cmd_force', 10)
        
        self.left_force = 0.0
        self.right_force = 0.0
        
        self.get_logger().info('Simple Robot Driver Started!')
        self.get_logger().info('W/S: Forward/Back, A/D: Turn, Space: Stop')
    
    def publish_forces(self):
        left_msg = Float64()
        right_msg = Float64()
        left_msg.data = self.left_force
        right_msg.data = self.right_force
        self.left_pub.publish(left_msg)
        self.right_pub.publish(right_msg)

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

def main():
    rclpy.init()
    node = SimpleRobotDriver()
    
    force_step = 0.5
    max_force = 2.0  # Reduced from 10.0 to prevent flipping
    
    try:
        while rclpy.ok():
            key = get_key()
            
            if key == 'w':  # Forward
                node.left_force = max_force
                node.right_force = max_force
                print(f"Forward: {node.left_force:.1f}", end='\r')
            elif key == 's':  # Backward
                node.left_force = -max_force
                node.right_force = -max_force
                print(f"Backward: {node.left_force:.1f}", end='\r')
            elif key == 'a':  # Turn left
                node.left_force = -max_force
                node.right_force = max_force
                print("Turn Left", end='\r')
            elif key == 'd':  # Turn right
                node.left_force = max_force
                node.right_force = -max_force
                print("Turn Right", end='\r')
            elif key == ' ':  # Stop
                node.left_force = 0.0
                node.right_force = 0.0
                print("STOP          ", end='\r')
            elif key == '\x03':  # Ctrl+C
                break
            
            node.publish_forces()
    
    except Exception as e:
        print(e)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

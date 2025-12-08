#!/usr/bin/env python3
"""
Simple LiDAR Tester
Run this while Gazebo is active to see if the robot detects obstacles.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import sys

class LidarTester(Node):
    def __init__(self):
        super().__init__('lidar_tester')
        self.subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.listener_callback,
            10)
        print("Listening for LiDAR data on /scan...")

    def listener_callback(self, msg):
        # Scan details
        num_readings = len(msg.ranges)
        min_dist = min(msg.ranges)
        max_dist = max([r for r in msg.ranges if r < float('inf')])
        
        # Center beam (straight ahead)
        center_idx = num_readings // 2
        center_dist = msg.ranges[center_idx]
        
        # Output status
        print(f"\r[LiDAR] Min: {min_dist:.2f}m | Max: {max_dist:.2f}m | Center: {center_dist:.2f}m | Rays: {num_readings}", end="")
        sys.stdout.flush()

def main(args=None):
    rclpy.init(args=args)
    tester = LidarTester()
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

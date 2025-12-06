#!/usr/bin/env python3
"""
Autonomous Lane Following Controller
Combines lane detection and obstacle detection for autonomous driving
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist, Point
from vision_msgs.msg import Detection2DArray
import time


class LaneFollower(Node):
    """ROS 2 Node for autonomous lane following with obstacle avoidance"""

    def __init__(self):
        super().__init__('lane_follower')

        # Parameters
        self.declare_parameter('target_speed', 2.0)  # m/s
        self.declare_parameter('steering_gain', 1.5)  # Steering sensitivity
        self.declare_parameter('obstacle_stop_distance', 5.0)  # meters
        self.declare_parameter('lane_loss_timeout', 2.0)  # seconds
        self.declare_parameter('enable_obstacle_avoidance', True)

        self.target_speed = self.get_parameter('target_speed').value
        self.steering_gain = self.get_parameter('steering_gain').value
        self.obstacle_distance = self.get_parameter('obstacle_stop_distance').value
        self.lane_timeout = self.get_parameter('lane_loss_timeout').value
        self.enable_obstacles = self.get_parameter('enable_obstacle_avoidance').value

        # State variables
        self.current_steering_angle = 0.0
        self.last_lane_time = time.time()
        self.obstacle_detected = False
        self.lane_center = None

        # Subscribers
        self.steering_sub = self.create_subscription(
            Float32,
            '/lane_steering_angle',
            self.steering_callback,
            10
        )

        self.lane_center_sub = self.create_subscription(
            Point,
            '/lane_center',
            self.lane_center_callback,
            10
        )

        if self.enable_obstacles:
            self.detection_sub = self.create_subscription(
                Detection2DArray,
                '/detections',
                self.detection_callback,
                10
            )

        # Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Control loop timer (20 Hz)
        self.control_timer = self.create_timer(0.05, self.control_loop)

        self.get_logger().info('Lane follower initialized')
        self.get_logger().info(f'Target speed: {self.target_speed} m/s')
        self.get_logger().info(f'Steering gain: {self.steering_gain}')
        self.get_logger().info(f'Obstacle avoidance: {self.enable_obstacles}')

    def steering_callback(self, msg: Float32):
        """Update steering angle from lane detection"""
        self.current_steering_angle = msg.data
        self.last_lane_time = time.time()

    def lane_center_callback(self, msg: Point):
        """Update lane center position"""
        self.lane_center = msg

    def detection_callback(self, msg: Detection2DArray):
        """Process obstacle detections"""
        if not self.enable_obstacles:
            return

        # Simple logic: stop if any object detected ahead
        # In production, you'd use depth/distance estimation
        if len(msg.detections) > 0:
            self.obstacle_detected = True
            self.get_logger().warn(
                f'Obstacle detected! {len(msg.detections)} objects',
                throttle_duration_sec=2
            )
        else:
            self.obstacle_detected = False

    def control_loop(self):
        """Main control loop - publishes velocity commands"""
        cmd = Twist()

        # Check if lane was recently detected
        time_since_lane = time.time() - self.last_lane_time
        lane_detected = time_since_lane < self.lane_timeout

        if not lane_detected:
            # Lost lane - emergency stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().warn(
                f'Lane lost for {time_since_lane:.1f}s - STOPPED',
                throttle_duration_sec=1
            )
        elif self.obstacle_detected:
            # Obstacle ahead - stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().warn('Obstacle ahead - STOPPED', throttle_duration_sec=1)
        else:
            # Normal operation - follow lane
            cmd.linear.x = self.target_speed
            cmd.angular.z = -self.current_steering_angle * self.steering_gain

            # Log status occasionally
            if int(time.time() * 2) % 10 == 0:  # Every 5 seconds
                self.get_logger().info(
                    f'Driving | Speed: {cmd.linear.x:.2f} m/s | '
                    f'Steering: {self.current_steering_angle:.3f} | '
                    f'Angular: {cmd.angular.z:.3f}',
                    throttle_duration_sec=5
                )

        # Publish command
        self.cmd_vel_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down lane follower')
    finally:
        # Send stop command
        stop_cmd = Twist()
        node.cmd_vel_pub.publish(stop_cmd)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

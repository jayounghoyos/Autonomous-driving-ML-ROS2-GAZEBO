#!/usr/bin/env python3
"""
Vision-Based Goal-Seeking Environment (Refactored)
Separates ROS interface from Gym environment logic.
Uses centralized config.
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2
import time

from config import RobotConfig, RewardConfig

class RobotInterface(Node):
    """
    Handles all ROS 2 communication.
    Think of this as the "Driver" for the robot.
    """
    def __init__(self):
        super().__init__('robot_interface')
        
        # Subscriptions
        self.create_subscription(Odometry, RobotConfig.TOPIC_ODOM, self._odom_callback, 10)
        self.create_subscription(Image, RobotConfig.TOPIC_IMAGE, self._camera_callback, 10)
        self.create_subscription(LaserScan, RobotConfig.TOPIC_SCAN, self._scan_callback, 10)
        
        # Ackermann Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, RobotConfig.TOPIC_CMD_VEL, 10)
        
        # Simulation Control
        self.goal_pub = self.create_publisher(Pose, RobotConfig.TOPIC_GOAL_POSE, 10)
        
        # State Data
        self.bridge = CvBridge()
        self.camera_image = None
        self.odom_data = None
        self.scan_data = None
        
        self.get_logger().info('RobotInterface initialized')

    def _odom_callback(self, msg):
        self.odom_data = msg

    def _camera_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            self.camera_image = cv2.resize(cv_image, 
                                         (RobotConfig.IMAGE_WIDTH, RobotConfig.IMAGE_HEIGHT))
        except Exception as e:
            self.get_logger().error(f'Camera callback error: {e}')

    def _scan_callback(self, msg):
        self.scan_data = msg

    def publish_action(self, action):
        """
        Convert action to Twist using Config Limits
        Action: [steering, velocity]
        """
        steer = float(np.clip(action[0], RobotConfig.MIN_STEER, RobotConfig.MAX_STEER))
        vel = float(np.clip(action[1], RobotConfig.MIN_VELOCITY, RobotConfig.MAX_VELOCITY))
        
        msg = Twist()
        msg.linear.x = vel
        msg.angular.z = steer
        
        # DEBUG: Only print every 10th command to avoid spam
        if np.random.rand() < 0.1:
           print(f"CMD: v={vel:.2f}, w={steer:.2f}")
        
        self.cmd_vel_pub.publish(msg)

    def move_goal(self, x, y):
        """Teleport the visual goal marker"""
        msg = Pose()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = 0.3 # Keep at original height
        self.goal_pub.publish(msg)

    def get_observation_data(self):
        """Return raw data packet"""
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.camera_image, self.odom_data, self.scan_data


class VisionGoalEnv(gym.Env):
    """
    The 'Brain' of the system.
    Standard Gymnasium interface for RL training.
    """
    def __init__(self):
        super().__init__()
        
        # Connect to ROS
        if not rclpy.ok():
            rclpy.init()
        self.robot = RobotInterface()
        
        # Action Space: [steering, velocity]
        self.action_space = spaces.Box(
            low=np.array([RobotConfig.MIN_STEER, RobotConfig.MIN_VELOCITY]),
            high=np.array([RobotConfig.MAX_STEER, RobotConfig.MAX_VELOCITY]),
            shape=(2,),
            dtype=np.float32
        )
        
        # Observation Space: Image + Vector
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, high=255,
                shape=(RobotConfig.IMAGE_HEIGHT, RobotConfig.IMAGE_WIDTH, 3),
                dtype=np.uint8
            ),
            'lidar': spaces.Box(
                low=0.0, high=10.0, shape=(RewardConfig.LIDAR_RAYS,), dtype=np.float32
            ),
            'vector': spaces.Box(
                low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
            )
        })
        
        self.goal_pos = np.array([5.0, 0.0])
        self.prev_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize goal
        self.goal_pos = np.array([
            np.random.uniform(*RewardConfig.GOAL_X_RANGE),
            np.random.uniform(*RewardConfig.GOAL_Y_RANGE)
        ])
        
        # Move the visual marker in Gazebo
        self.robot.move_goal(self.goal_pos[0], self.goal_pos[1])
        
        obs = self._get_observation()
        self.prev_distance = obs['vector'][0]
        return obs, {}

    def step(self, action):
        # 1. Action
        self.robot.publish_action(action)
        
        # 2. Observation
        obs = self._get_observation()
        
        # 3. Processing
        green_vals = self._process_vision(obs['image'])
        lidar_vals = obs['lidar']
        vec_vals = obs['vector'] # [dist, angle, lin_vel, ang_vel]
        
        # Get current position (re-extract from robot data or vector logic if passed)
        # Better to pass position directly from step or _get_observation if available
        # For now, we'll access the latest odom data via self.robot.odom_data (not ideal but quick)
        # OR: We can augment _calculate_reward to accept position.
        
        pos = self._get_current_pos()
        reward, terminated = self._calculate_reward(green_vals, vec_vals, lidar_vals, action, pos)
        
        # Visualization
        self._visualize(obs['image'], reward, green_vals)
        
        return obs, reward, terminated, False, {}

    def _get_observation(self):
        # Spin until we get data
        image, odom, scan = None, None, None
        for _ in range(50):
            image, odom, scan = self.robot.get_observation_data()
            if image is not None and odom is not None and scan is not None:
                break
        
        # Fallbacks
        if image is None: 
            image = np.zeros((RobotConfig.IMAGE_HEIGHT, RobotConfig.IMAGE_WIDTH, 3), dtype=np.uint8)
        
        # Process Vector Data
        if odom:
            pos = odom.pose.pose.position
            current_pos = np.array([pos.x, pos.y])
            
            # Distance
            goal_vec = self.goal_pos - current_pos
            distance = np.linalg.norm(goal_vec)
            
            # Angle
            # ... (Simplified quaternion math for brevity)
            q = odom.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            global_angle = np.arctan2(goal_vec[1], goal_vec[0])
            rel_angle = global_angle - yaw
            # Normalize -pi to pi
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            
            lin_vel = odom.twist.twist.linear.x
            ang_vel = odom.twist.twist.angular.z
        else:
            distance, rel_angle, lin_vel, ang_vel = 5.0, 0.0, 0.0, 0.0
            
        vector = np.array([distance, rel_angle, lin_vel, ang_vel], dtype=np.float32)
        lidar_processed = self._process_lidar(scan)
        
        return {'image': image, 'vector': vector, 'lidar': lidar_processed}

    def _process_lidar(self, scan):
        """Downsample 360 rays to LIDAR_RAYS"""
        if scan is None:
            return np.ones(RewardConfig.LIDAR_RAYS, dtype=np.float32) * 10.0
            
        ranges = np.array(scan.ranges)
        ranges[ranges == 0] = 10.0 # Fix zeros
        ranges[np.isinf(ranges)] = 10.0 # Fix infs
        
        # Binning (Min pooling for safety)
        bins = np.array_split(ranges, RewardConfig.LIDAR_RAYS)
        processed = np.array([np.min(b) for b in bins], dtype=np.float32)
        return processed

    def _process_vision(self, image):
        """Detect green goal. Returns (area_ratio, center_x)"""
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array([35, 100, 100]), np.array([85, 255, 255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest) / (RobotConfig.IMAGE_WIDTH * RobotConfig.IMAGE_HEIGHT)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                return area, cx / RobotConfig.IMAGE_WIDTH
        
        return 0.0, 0.5 # Default: No area, center frame

    def _get_current_pos(self):
        """Helper to safely get position"""
        if self.robot.odom_data:
            p = self.robot.odom_data.pose.pose.position
            return np.array([p.x, p.y])
        return np.array([0.0, 0.0])

    def _calculate_reward(self, green_vals, vec_vals, lidar_vals, action, current_pos):
        """Pure reward logic using Config"""
        area, center_x = green_vals
        distance, angle, lin_vel, ang_vel = vec_vals
        
        reward = 0.0
        terminated = False
        
        # 0. Geo-fence Check
        x, y = current_pos
        if not (RewardConfig.BOUNDARY_X[0] < x < RewardConfig.BOUNDARY_X[1]) or \
           not (RewardConfig.BOUNDARY_Y[0] < y < RewardConfig.BOUNDARY_Y[1]):
            return -RewardConfig.OUT_OF_BOUNDS_PENALTY, True
            
        # Safety: Collision Detection
        if np.min(lidar_vals) < RewardConfig.COLLISION_DISTANCE:
            return -RewardConfig.COLLISION_PENALTY, True
        
        # Success Logic
        if distance < RewardConfig.GOAL_DISTANCE_THRESHOLD or area > RewardConfig.GOAL_VISION_THRESHOLD:
            return RewardConfig.GOAL_REACHED_BONUS, True
            
        # 1. Progress
        reward += (self.prev_distance - distance) * RewardConfig.DISTANCE_IMPROVEMENT_MULTIPLIER
        
        # 2. Vision
        if area > 0.01:
            reward += area * RewardConfig.VISION_VISIBLE_BONUS
            reward += (0.5 - abs(center_x - 0.5)) * RewardConfig.VISION_CENTER_BONUS
        else:
            reward -= abs(angle) * RewardConfig.HEADING_PENALTY_WEIGHT # GPS Guidance
            reward -= RewardConfig.BLIND_PENALTY
            
        # 3. Penalties
        reward -= RewardConfig.TIME_PENALTY
        if abs(ang_vel) > 1.5: reward -= abs(ang_vel) * RewardConfig.SPIN_PENALTY_WEIGHT
        if lin_vel < -0.1: reward -= abs(lin_vel) * RewardConfig.REVERSE_PENALTY_WEIGHT
        
        # 4. Stuck Detection
        # If trying to move (action > 0.5) but velocity is low (< 0.01)
        if np.mean(np.abs(action)) > 0.5 and abs(lin_vel) < RewardConfig.STUCK_VELOCITY_THRESHOLD:
            reward -= RewardConfig.STUCK_PENALTY_WEIGHT
        
        self.prev_distance = distance
        return reward, terminated

    def _visualize(self, image, reward, green_vals):
        """Debug GUI"""
        viz = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        viz = cv2.resize(viz, (300, 300), interpolation=cv2.INTER_NEAREST)
        cv2.putText(viz, f"R: {reward:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("Agent Brain", viz)
        cv2.waitKey(1)
    
    def close(self):
        self.robot.destroy_node()


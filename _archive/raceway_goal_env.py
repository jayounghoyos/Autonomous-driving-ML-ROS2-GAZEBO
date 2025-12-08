#!/usr/bin/env python3
"""
Raceway Navigation Environment
Waypoint-based goal seeking on Sonoma Raceway
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Pose, Point, Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
import cv2

from config import RobotConfig, RewardConfig

class RobotInterface(Node):
    """Handles ROS communication for raceway robot"""
    def __init__(self):
        super().__init__('robot_interface')
        
        # Subscriptions
        self.create_subscription(Odometry, RobotConfig.TOPIC_ODOM, self._odom_callback, 10)
        self.create_subscription(Image, RobotConfig.TOPIC_IMAGE, self._camera_callback, 10)
        self.create_subscription(LaserScan, RobotConfig.TOPIC_SCAN, self._scan_callback, 10)
        
        # Ackermann Publisher
        self.cmd_vel_pub = self.create_publisher(Twist, RobotConfig.TOPIC_CMD_VEL, 10)
        
        # State Data
        self.bridge = CvBridge()
        self.camera_image = None
        self.odom_data = None
        self.scan_data = None
        
        self.get_logger().info('Raceway RobotInterface initialized')

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
        """Convert action to Twist"""
        steer = float(np.clip(action[0], RobotConfig.MIN_STEER, RobotConfig.MAX_STEER))
        vel = float(np.clip(action[1], RobotConfig.MIN_VELOCITY, RobotConfig.MAX_VELOCITY))
        
        msg = Twist()
        msg.linear.x = vel
        msg.angular.z = steer
        self.cmd_vel_pub.publish(msg)

    def get_observation_data(self):
        """Return raw sensor data"""
        rclpy.spin_once(self, timeout_sec=0.1)
        return self.camera_image, self.odom_data, self.scan_data


class RacewayGoalEnv(gym.Env):
    """
    Waypoint-based navigation environment for racing
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
        
        # Observation Space: Image + LiDAR + Vector
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
                low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
            )  # [dist_to_waypoint, angle_to_waypoint, lin_vel, ang_vel, waypoint_idx, lap_progress]
        })
        
        # Waypoint tracking
        self.waypoints = RewardConfig.WAYPOINTS
        self.current_waypoint_idx = 0
        self.laps_completed = 0
        self.prev_distance = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset waypoint tracking
        self.current_waypoint_idx = 0
        self.laps_completed = 0
        
        obs = self._get_observation()
        self.prev_distance = obs['vector'][0]
        return obs, {}

    def step(self, action):
        # 1. Execute action
        self.robot.publish_action(action)
        
        # 2. Get observation
        obs = self._get_observation()
        
        # 3. Extract data
        lidar_vals = obs['lidar']
        vec_vals = obs['vector']
        
        # 4. Calculate reward
        pos = self._get_current_pos()
        reward, terminated = self._calculate_reward(vec_vals, lidar_vals, action, pos)
        
        # 5. Visualize
        self._visualize(obs['image'], reward, vec_vals)
        
        return obs, reward, terminated, False, {}

    def _get_observation(self):
        image, odom, scan = None, None, None
        for _ in range(50):
            image, odom, scan = self.robot.get_observation_data()
            if image is not None and odom is not None and scan is not None:
                break
        
        # Fallbacks
        if image is None: 
            image = np.zeros((RobotConfig.IMAGE_HEIGHT, RobotConfig.IMAGE_WIDTH, 3), dtype=np.uint8)
        
        # Process odometry and waypoints
        if odom:
            pos = odom.pose.pose.position
            current_pos = np.array([pos.x, pos.y])
            
            # Get current waypoint
            target_waypoint = self.waypoints[self.current_waypoint_idx]
            
            # Distance and angle to waypoint
            goal_vec = np.array(target_waypoint) - current_pos
            distance = np.linalg.norm(goal_vec)
            
            # Check if waypoint reached
            if distance < RewardConfig.WAYPOINT_RADIUS:
                self.current_waypoint_idx += 1
                if self.current_waypoint_idx >= len(self.waypoints):
                    self.laps_completed += 1
                    self.current_waypoint_idx = 0
            
            # Angle calculation
            q = odom.pose.pose.orientation
            siny_cosp = 2 * (q.w * q.z + q.x * q.y)
            cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
            yaw = np.arctan2(siny_cosp, cosy_cosp)
            
            global_angle = np.arctan2(goal_vec[1], goal_vec[0])
            rel_angle = global_angle - yaw
            rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi
            
            lin_vel = odom.twist.twist.linear.x
            ang_vel = odom.twist.twist.angular.z
        else:
            distance, rel_angle, lin_vel, ang_vel = 5.0, 0.0, 0.0, 0.0
        
        # Calculate lap progress (0-1)        
        lap_progress = self.current_waypoint_idx / len(self.waypoints)
        
        vector = np.array([
            distance, rel_angle, lin_vel, ang_vel, 
            float(self.current_waypoint_idx), lap_progress
        ], dtype=np.float32)
        
        lidar_processed = self._process_lidar(scan)
        
        return {'image': image, 'vector': vector, 'lidar': lidar_processed}

    def _process_lidar(self, scan):
        """Downsample LiDAR data"""
        if scan is None:
            return np.ones(RewardConfig.LIDAR_RAYS, dtype=np.float32) * 10.0
            
        ranges = np.array(scan.ranges)
        ranges[ranges == 0] = 10.0
        ranges[np.isinf(ranges)] = 10.0
        
        bins = np.array_split(ranges, RewardConfig.LIDAR_RAYS)
        processed = np.array([np.min(b) for b in bins], dtype=np.float32)
        return processed

    def _get_current_pos(self):
        """Get robot position"""
        if self.robot.odom_data:
            p = self.robot.odom_data.pose.pose.position
            return np.array([p.x, p.y])
        return np.array([0.0, 0.0])

    def _calculate_reward(self, vec_vals, lidar_vals, action, current_pos):
        """Waypoint-based reward function"""
        distance, angle, lin_vel, ang_vel, waypoint_idx, lap_progress = vec_vals
        
        reward = 0.0
        terminated = False
        
        # 0. Boundary check
        x, y = current_pos
        if not (RewardConfig.BOUNDARY_X[0] < x < RewardConfig.BOUNDARY_X[1]) or \
           not (RewardConfig.BOUNDARY_Y[0] < y < RewardConfig.BOUNDARY_Y[1]):
            return -RewardConfig.OUT_OF_BOUNDS_PENALTY, True
            
        # 1. Collision Detection
        if np.min(lidar_vals) < RewardConfig.COLLISION_DISTANCE:
            return -RewardConfig.COLLISION_PENALTY, True
        
        # 2. Waypoint Progress Reward
        dist_improvement = self.prev_distance - distance
        reward += dist_improvement * RewardConfig.DISTANCE_IMPROVEMENT_MULTIPLIER
        
        # 3. Waypoint Reached Bonus
        if distance < RewardConfig.WAYPOINT_RADIUS and dist_improvement < 0:
            # Just reached waypoint
            reward += RewardConfig.WAYPOINT_REWARD
        
        # 4. Lap Completion Bonus
        if self.laps_completed > 0:
            reward += RewardConfig.LAP_COMPLETION_BONUS
            self.laps_completed = 0  # Reset to avoid double rewards
        
        # 5. Forward progress incentive
        reward += lin_vel * 0.1  # Small reward for moving forward
        
        # 6. Penalties
        reward -= RewardConfig.TIME_PENALTY
        if abs(ang_vel) > 1.5: 
            reward -= abs(ang_vel) * RewardConfig.SPIN_PENALTY_WEIGHT
        
        # 7. Smooth steering penalty
        if abs(action[0]) > 0.6:
            reward -= 0.1  # Penalize aggressive steering
        
        self.prev_distance = distance
        return reward, terminated

    def _visualize(self, image, reward, vec_vals):
        """Debug visualization"""
        viz = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        viz = cv2.resize(viz, (300, 300), interpolation=cv2.INTER_NEAREST)
        
        dist = vec_vals[0]
        waypoint_idx = int(vec_vals[4])
        lap_progress = vec_vals[5]
        
        cv2.putText(viz, f"R: {reward:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(viz, f"WP: {waypoint_idx}/{len(self.waypoints)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(viz, f"Prog: {lap_progress:.1%}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        cv2.imshow("Raceway Agent", viz)
        cv2.waitKey(1)
    
    def close(self):
        self.robot.destroy_node()

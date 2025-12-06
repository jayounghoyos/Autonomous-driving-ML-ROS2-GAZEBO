#!/usr/bin/env python3
"""
Gym Environment for Gazebo-based RL Training
Bridges ROS 2 Gazebo simulation with OpenAI Gym for PPO training
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import math
import time


class GazeboNavigationEnv(gym.Env, Node):
    """
    Custom Gym Environment for obstacle avoidance with goal navigation

    Observation Space:
        - LiDAR readings (360 values, 0-10m range)
        - Goal position relative to robot (distance, angle)
        - Current velocity (linear, angular)
        Total: 363 dimensions

    Action Space:
        - Linear velocity: [-1.0, 1.0] m/s
        - Angular velocity: [-1.0, 1.0] rad/s

    Reward Function:
        +100: Reaching goal
        -50: Collision
        +distance_reduced: Moving toward goal
        -0.1: Time penalty (encourages efficiency)
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, namespace='', goal_tolerance=0.5):
        gym.Env.__init__(self)
        Node.__init__(self, 'gazebo_nav_env_' + namespace.replace('/', '_'))

        self.namespace = namespace
        self.goal_tolerance = goal_tolerance

        # Environment state
        self.lidar_data = None
        self.odom_data = None
        self.current_pos = np.array([0.0, 0.0])
        self.current_yaw = 0.0
        self.goal_pos = np.array([10.0, 10.0])  # Will be randomized
        self.prev_distance_to_goal = None
        self.collision = False
        self.goal_reached = False

        # Define action and observation spaces
        # Actions: [linear_vel, angular_vel]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observations: [lidar_360, goal_distance, goal_angle, lin_vel, ang_vel]
        self.observation_space = spaces.Box(
            low=np.concatenate([
                np.zeros(360),  # LiDAR readings
                np.array([0.0, -np.pi, -3.0, -2.0])  # goal_dist, goal_angle, velocities
            ]),
            high=np.concatenate([
                np.full(360, 10.0),  # LiDAR max range
                np.array([50.0, np.pi, 3.0, 2.0])  # max values
            ]),
            dtype=np.float32
        )

        # ROS 2 setup
        self.bridge_process = None
        self.setup_ros_interfaces()
        
        # Spawn robot and bridge
        self.spawn_robot()
        self.spawn_bridge()

        # Timing
        self.episode_start_time = time.time()
        self.max_episode_time = 60.0  # seconds

    def spawn_bridge(self):
        """Spawn ROS-Gazebo bridge for this robot"""
        import subprocess
        
        robot_name = self.namespace.replace('/', '')
        
        # Topic names in Gazebo
        gz_cmd_vel = f"/model/{robot_name}/cmd_vel"
        gz_odom = f"/model/{robot_name}/odometry"
        
        # Explicit direction to prevent loops and confusion
        # ] = ROS -> Gazebo (Subscriber on ROS, Publisher on Gz)  <-- For cmd_vel
        # [ = Gazebo -> ROS (Subscriber on Gz, Publisher on ROS)  <-- For Odometry
        
        # SYNTAX CRITICAL: First separator is ALWAYS @, second is direction.
        # Format: topic@ROS_type]GZ_type
        
        args = []
        # cmd_vel: ROS -> Gz
        args.append(f"{gz_cmd_vel}@geometry_msgs/msg/Twist]gz.msgs.Twist")
        # odom: Gz -> ROS
        args.append(f"{gz_odom}@nav_msgs/msg/Odometry[gz.msgs.Odometry")
        
        cmd = ["ros2", "run", "ros_gz_bridge", "parameter_bridge"] + args + [
            "--ros-args",
            "-r", f"{gz_cmd_vel}:={self.namespace}/cmd_vel",
            "-r", f"{gz_odom}:={self.namespace}/odom"
        ]
        
        self.get_logger().info(f'Starting bridge: {" ".join(cmd)}')
        self.bridge_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def spawn_robot(self):
        """Spawn robot in Gazebo"""
        import subprocess
        import os
        
        # Calculate spawn position based on rank (to avoid collision)
        try:
            rank = int(self.namespace.split('_')[-1])
        except (ValueError, IndexError):
            rank = 0
            
        # Grid layout for spawn
        row = rank // 3
        col = rank % 3
        x_pos = -10.0 + (col * 3.0)  # Increased spacing
        y_pos = -10.0 + (row * 3.0)
        
        model_path = "/home/jayoungh/PersonalPorjects/Autonomous-driving-ML-ROS2-GAZEBO/src/vehicle_description/urdf/rl_training_car.sdf"
        
        # We need to make the SDF unique per robot to get unique topics?
        # Or we can use the <ros> <namespace> plugin in SDF? 
        # But Gz Sim doesn't use <ros> tag natively like Classic.
        
        cmd = [
            "ros2", "run", "ros_gz_sim", "create",
            "-world", "rl_training_world",
            "-file", model_path,
            "-name", self.namespace.replace('/', ''),
            "-x", str(x_pos),
            "-y", str(y_pos),
            "-z", "0.5"
        ]
        
        # Use simple subprocess instead of Popen to ensure it completes before next?
        # No, create is async service call usually.
        subprocess.Popen(cmd)
        time.sleep(2.0)

    def delete_robot(self):
        """Delete robot from Gazebo"""
        import subprocess
        cmd = [
            "ros2", "run", "ros_gz_sim", "remove",
            "-world", "rl_training_world",
            "-name", self.namespace.replace('/', '')
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def setup_ros_interfaces(self):
        """Setup ROS 2 publishers and subscribers"""
        topic_prefix = self.namespace if self.namespace else ''

        # Subscribers
        self.scan_sub = self.create_subscription(
            LaserScan,
            f'{topic_prefix}/scan',
            self.lidar_callback,
            10
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            f'{topic_prefix}/odom',
            self.odom_callback,
            10
        )

        # Publishers
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            f'{topic_prefix}/cmd_vel',
            10
        )

    def lidar_callback(self, msg):
        """Process LiDAR scan data"""
        ranges = np.array(msg.ranges)
        # Replace inf with max range
        ranges[np.isinf(ranges)] = msg.range_max
        self.lidar_data = ranges

        # Check for collision (obstacle within 0.3m)
        if np.min(ranges) < 0.3:
            self.collision = True

    def odom_callback(self, msg):
        """Process odometry data"""
        self.odom_data = msg

        # Extract position
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ])

        # Extract orientation (yaw)
        orientation = msg.pose.pose.orientation
        siny_cosp = 2 * (orientation.w * orientation.z + orientation.x * orientation.y)
        cosy_cosp = 1 - 2 * (orientation.y**2 + orientation.z**2)
        self.current_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Check if goal reached
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)
        if distance_to_goal < self.goal_tolerance:
            self.goal_reached = True

    def get_observation(self):
        """Construct observation vector"""
        # Spin ROS to update data
        rclpy.spin_once(self, timeout_sec=0.01)

        if self.lidar_data is None or self.odom_data is None:
            # Return zero observation if data not yet available
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Calculate goal relative to robot
        goal_vector = self.goal_pos - self.current_pos
        distance_to_goal = np.linalg.norm(goal_vector)
        angle_to_goal = math.atan2(goal_vector[1], goal_vector[0]) - self.current_yaw

        # Normalize angle to [-pi, pi]
        angle_to_goal = math.atan2(math.sin(angle_to_goal), math.cos(angle_to_goal))

        # Get current velocities
        lin_vel = self.odom_data.twist.twist.linear.x
        ang_vel = self.odom_data.twist.twist.angular.z

        # Construct observation
        observation = np.concatenate([
            self.lidar_data.astype(np.float32),
            np.array([distance_to_goal, angle_to_goal, lin_vel, ang_vel], dtype=np.float32)
        ])

        return observation

    def step(self, action):
        """Execute action and return observation, reward, done, info"""
        # Publish action
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_vel_pub.publish(cmd)

        # Wait for physics to update (100 Hz control loop)
        time.sleep(0.01)

        # Get new observation
        observation = self.get_observation()

        # Calculate reward
        reward, done, info = self.calculate_reward()

        truncated = (time.time() - self.episode_start_time) > self.max_episode_time

        return observation, reward, done, truncated, info

    def calculate_reward(self):
        """Calculate reward based on current state"""
        reward = 0.0
        done = False
        info = {}

        # Goal reached reward
        if self.goal_reached:
            reward = 100.0
            done = True
            info['success'] = True
            self.get_logger().info('🎯 Goal reached!')
            return reward, done, info

        # Collision penalty
        if self.collision:
            reward = -50.0
            done = True
            info['collision'] = True
            self.get_logger().info('💥 Collision!')
            return reward, done, info

        # Progress toward goal reward
        distance_to_goal = np.linalg.norm(self.current_pos - self.goal_pos)

        if self.prev_distance_to_goal is not None:
            distance_reduced = self.prev_distance_to_goal - distance_to_goal
            reward += distance_reduced * 1.0  # Reward for getting closer

        self.prev_distance_to_goal = distance_to_goal

        # Time penalty (encourages efficiency)
        reward -= 0.1

        # Proximity penalty (discourage getting too close to obstacles)
        if self.lidar_data is not None:
            min_distance = np.min(self.lidar_data)
            if min_distance < 1.0:
                reward -= (1.0 - min_distance) * 0.5

        return reward, done, info

    def reset(self, seed=None, options=None):
        """Reset environment to start new episode"""
        super().reset(seed=seed)

        # Stop the robot
        cmd = Twist()
        self.cmd_vel_pub.publish(cmd)

        # Reset state
        self.collision = False
        self.goal_reached = False
        self.prev_distance_to_goal = None
        self.episode_start_time = time.time()

        # Randomize goal position (within training area)
        if seed is not None:
            np.random.seed(seed)

        self.goal_pos = np.array([
            np.random.uniform(-12, 12),
            np.random.uniform(-12, 12)
        ])
        
        self.reset_robot_pose()

        # For now, wait for simulation to stabilize
        time.sleep(0.5)

        observation = self.get_observation()
        info = {}

        return observation, info

    def reset_robot_pose(self):
        """Reset robot pose to a safe start location"""
        pass

    def render(self, mode='human'):
        """Render environment (handled by Gazebo GUI)"""
        pass

    def close(self):
        """Cleanup ROS resources"""
        if self.bridge_process:
            self.bridge_process.kill()
        self.delete_robot()
        self.destroy_node()

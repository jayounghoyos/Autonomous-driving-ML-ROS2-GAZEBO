#!/usr/bin/env python3
"""
Simplified RL Environment - Goal Seeking Only (No LiDAR)
Observation: goal distance, goal angle, linear velocity, angular velocity
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import math


class GoalSeekingEnv(gym.Env, Node):
    """
    Simple goal-seeking environment without obstacles
    
    Observation: [goal_distance, goal_angle, linear_vel, angular_vel] = 4 dims
    Action: [left_force, right_force] in [-2.0, 2.0]
    """
    
    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, 'goal_seeking_env')
        
        # Action: left and right wheel forces
        self.action_space = spaces.Box(
            low=np.array([-2.0, -2.0]),
            high=np.array([2.0, 2.0]),
            dtype=np.float32
        )
        
        # Observation: goal info + velocity
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.pi, -5.0, -5.0]),
            high=np.array([np.inf, np.pi, 5.0, 5.0]),
            dtype=np.float32
        )
        
        # State
        self.odom_data = None
        self.goal_pos = np.array([5.0, 0.0])
        self.prev_distance = None
        
        # Publishers
        self.left_force_pub = self.create_publisher(
            Float64, '/model/simple_robot/joint/left_wheel_joint/cmd_force', 10)
        self.right_force_pub = self.create_publisher(
            Float64, '/model/simple_robot/joint/right_wheel_joint/cmd_force', 10)
        
        # Subscribers
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        
        self.get_logger().info('GoalSeekingEnv initialized!')
    
    def _odom_callback(self, msg):
        self.odom_data = msg
    
    def _get_observation(self):
        # Wait for data
        for _ in range(50):
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.odom_data is not None:
                break
        
        if self.odom_data:
            pos = self.odom_data.pose.pose.position
            robot_pos = np.array([pos.x, pos.y])
            goal_vec = self.goal_pos - robot_pos
            distance = np.linalg.norm(goal_vec)
            angle = np.arctan2(goal_vec[1], goal_vec[0])
            
            # Get velocity
            lin_vel = self.odom_data.twist.twist.linear.x
            ang_vel = self.odom_data.twist.twist.angular.z
        else:
            distance, angle = 5.0, 0.0
            lin_vel, ang_vel = 0.0, 0.0
        
        obs = np.array([distance, angle, lin_vel, ang_vel], dtype=np.float32)
        return obs
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize goal
        angle = np.random.uniform(-np.pi, np.pi)
        dist = np.random.uniform(3.0, 7.0)
        self.goal_pos = np.array([dist * np.cos(angle), dist * np.sin(angle)])
        
        self.prev_distance = None
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        # Apply forces
        left_msg = Float64()
        right_msg = Float64()
        left_msg.data = float(action[0])
        right_msg.data = float(action[1])
        self.left_force_pub.publish(left_msg)
        self.right_force_pub.publish(right_msg)
        
        # Wait for physics
        rclpy.spin_once(self, timeout_sec=0.1)
        
        obs = self._get_observation()
        distance = obs[0]
        
        reward = 0.0
        terminated = False
        
        # Goal reached
        if distance < 0.5:
            reward = 100.0
            terminated = True
            self.get_logger().info('🎯 Goal reached!')
        
        # Progress reward
        elif self.prev_distance is not None:
            reward = (self.prev_distance - distance) * 10.0
        
        self.prev_distance = distance
        reward -= 0.1  # Time penalty
        
        return obs, reward, terminated, False, {}
    
    def close(self):
        self.destroy_node()


def main():
    """Test the environment"""
    rclpy.init()
    env = GoalSeekingEnv()
    
    print("Testing Goal-Seeking Environment...")
    obs, info = env.reset()
    print(f"Observation: distance={obs[0]:.2f}, angle={obs[1]:.2f}, vel=({obs[2]:.2f}, {obs[3]:.2f})")
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, distance={obs[0]:.2f}, done={done}")
        if done:
            print(f"Episode finished at step {i}")
            obs, info = env.reset()
    
    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

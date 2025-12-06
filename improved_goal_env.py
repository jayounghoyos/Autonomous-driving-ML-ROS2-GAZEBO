#!/usr/bin/env python3
"""
Improved Goal-Seeking Environment with Better Reward Shaping
For 4-wheel differential drive robot
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64
from nav_msgs.msg import Odometry
import math


class ImprovedGoalSeekingEnv(gym.Env, Node):
    """
    Improved environment with better reward shaping
    
    Observation: [goal_distance, goal_angle, linear_vel, angular_vel] = 4 dims
    Action: [left_force, right_force] in [-2.0, 2.0]
    """
    
    def __init__(self):
        gym.Env.__init__(self)
        Node.__init__(self, 'improved_goal_env')
        
        # Action: left and right side forces (each side has 2 wheels)
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
        self.initial_distance = None
        
        # Publishers for 4 wheels (left side + right side)
        self.front_left_pub = self.create_publisher(
            Float64, '/model/four_wheel_robot/joint/front_left_wheel_joint/cmd_force', 10)
        self.rear_left_pub = self.create_publisher(
            Float64, '/model/four_wheel_robot/joint/rear_left_wheel_joint/cmd_force', 10)
        self.front_right_pub = self.create_publisher(
            Float64, '/model/four_wheel_robot/joint/front_right_wheel_joint/cmd_force', 10)
        self.rear_right_pub = self.create_publisher(
            Float64, '/model/four_wheel_robot/joint/rear_right_wheel_joint/cmd_force', 10)
        
        # Subscribers
        self.create_subscription(Odometry, '/odom', self._odom_callback, 10)
        
        self.get_logger().info('ImprovedGoalSeekingEnv initialized!')
    
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
        self.initial_distance = None
        
        obs = self._get_observation()
        self.initial_distance = obs[0]
        self.prev_distance = obs[0]
        
        return obs, {}
    
    def step(self, action):
        # Apply forces to all 4 wheels (left side + right side)
        left_msg = Float64()
        right_msg = Float64()
        left_msg.data = float(action[0])
        right_msg.data = float(action[1])
        
        # Left side (front + rear)
        self.front_left_pub.publish(left_msg)
        self.rear_left_pub.publish(left_msg)
        
        # Right side (front + rear)
        self.front_right_pub.publish(right_msg)
        self.rear_right_pub.publish(right_msg)
        
        # Wait for physics
        rclpy.spin_once(self, timeout_sec=0.1)
        
        obs = self._get_observation()
        distance = obs[0]
        angle = obs[1]
        
        # IMPROVED REWARD FUNCTION
        reward = 0.0
        terminated = False
        
        # 1. Goal reached - BIG reward
        if distance < 0.5:
            reward = 200.0
            terminated = True
            self.get_logger().info('Goal reached!')
        
        else:
            # 2. Distance-based reward (main signal)
            if self.prev_distance is not None:
                distance_reward = (self.prev_distance - distance) * 20.0  # Increased from 10.0
                reward += distance_reward
            
            # 3. Proximity bonus (encourages getting close)
            proximity_bonus = 10.0 / (distance + 1.0)
            reward += proximity_bonus
            
            # 4. STRONG Heading reward (face the goal) - INCREASED
            heading_reward = -abs(angle) * 10.0  # Increased from 2.0 to 10.0
            reward += heading_reward
            
            # 4b. Turning bonus (reward for reducing angle)
            if hasattr(self, 'prev_angle') and self.prev_angle is not None:
                angle_improvement = abs(self.prev_angle) - abs(angle)
                reward += angle_improvement * 15.0  # Big bonus for turning toward goal
            self.prev_angle = angle
            
            # 5. Progress reward (percentage of distance covered)
            if self.initial_distance is not None and self.initial_distance > 0:
                progress = (self.initial_distance - distance) / self.initial_distance
                reward += progress * 5.0
            
            # 6. Small time penalty (encourage efficiency, but not too harsh)
            reward -= 0.05
            
            # 7. Penalty for excessive spinning
            angular_vel = obs[3]
            if abs(angular_vel) > 1.0:
                reward -= abs(angular_vel) * 0.5
        
        self.prev_distance = distance
        
        return obs, reward, terminated, False, {}
    
    def close(self):
        self.destroy_node()


def main():
    """Test the environment"""
    rclpy.init()
    env = ImprovedGoalSeekingEnv()
    
    print("Testing Improved Environment...")
    obs, info = env.reset()
    print(f"Initial: distance={obs[0]:.2f}, angle={obs[1]:.2f}")
    
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step {i}: reward={reward:.2f}, distance={obs[0]:.2f}, done={done}")
        if done:
            print("Goal reached!")
            break
    
    env.close()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

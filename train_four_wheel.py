#!/usr/bin/env python3
"""
Training script for 4-wheel robot with improved rewards
"""
import os
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import argparse
sys.path.append(os.path.dirname(__file__))
from improved_goal_env import ImprovedGoalSeekingEnv


def make_env():
    """Create environment with ROS initialized"""
    def _init():
        if not rclpy.ok():
            rclpy.init()
        return ImprovedGoalSeekingEnv()
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps')
    args = parser.parse_args()
    
    print("Starting Training - 4-Wheel Robot with Improved Rewards")
    print("This uses the Gazebo GUI you have open!")
    
    # Create single environment
    env = DummyVecEnv([make_env()])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='four_wheel_improved'
    )
    
    # Create PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        verbose=1,
        device='cuda',
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log='./tensorboard_logs/'
    )
    
    print("\nTraining Configuration:")
    print(f"  - Robot: 4-Wheel Differential Drive")
    print(f"  - Reward: Improved 7-component shaping")
    print(f"  - Device: CUDA GPU")
    print(f"  - Total Timesteps: {args.timesteps:,}")
    print(f"  - Checkpoints: Every 10,000 steps")
    print("\nWatch the robot learn in your Gazebo window!\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        model.save("./models/four_wheel_improved_final")
        print("\nTraining complete! Model saved to ./models/four_wheel_improved_final")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        model.save("./models/four_wheel_improved_interrupted")
        print("Model saved to ./models/four_wheel_improved_interrupted")
    
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

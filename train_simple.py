#!/usr/bin/env python3
"""
Simplified PPO Training - Single Environment (Most Stable)
Use this for reliable training without parallel complexity
"""
import os
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import argparse
sys.path.append(os.path.dirname(__file__))
from goal_seeking_env import GoalSeekingEnv


def make_env():
    """Create environment with ROS initialized"""
    def _init():
        if not rclpy.ok():
            rclpy.init()
        return GoalSeekingEnv()
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps (default: 500k)')
    args = parser.parse_args()
    
    print("🚀 Starting PPO Training (Single Environment - Stable)")
    print("💡 This uses the Gazebo GUI you already have open!")
    
    # Create single environment
    env = DummyVecEnv([make_env()])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='goal_seeking_ppo'
    )
    
    # Create PPO agent with GPU
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
    
    print("\n📊 Training Configuration:")
    print(f"  - Device: CUDA (GPU) 🚀")
    print(f"  - Environments: 1 (using your open Gazebo GUI)")
    print(f"  - Observation Space: {env.observation_space}")
    print(f"  - Action Space: {env.action_space}")
    print(f"  - Total Timesteps: {args.timesteps:,}")
    print(f"  - Checkpoints: Every 10,000 steps")
    print("\n🎯 Goal: Robot learns to navigate to green sphere")
    print("👀 Watch the robot learn in your Gazebo window!\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        # Save final model
        model.save("./models/goal_seeking_final")
        print("\n✅ Training complete! Model saved to ./models/goal_seeking_final")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        model.save("./models/goal_seeking_interrupted")
        print("Model saved to ./models/goal_seeking_interrupted")
    
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

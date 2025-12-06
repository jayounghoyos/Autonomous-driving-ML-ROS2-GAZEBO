#!/usr/bin/env python3
"""
Parallel PPO Training for Gazebo Navigation
Trains multiple environments simultaneously for faster learning
"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import os
from pathlib import Path
import argparse

# Import our custom environment
from .gazebo_env import GazeboNavigationEnv


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env creation

    Args:
        rank: Index of the subprocess
        seed: Random seed
    """
    def _init():
        # Initialize ROS for this process
        if not rclpy.ok():
            rclpy.init()

        env = GazeboNavigationEnv(
            namespace=f'/robot_{rank}',
            goal_tolerance=0.5
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(
    total_timesteps=500000,
    n_envs=8,
    save_dir='models/ppo_navigation',
    log_dir='logs/ppo_navigation',
    eval_freq=10000,
    save_freq=50000,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    device='cuda'
):
    """
    Train PPO agent with parallel environments

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        log_dir: TensorBoard log directory
        eval_freq: Evaluate agent every N steps
        save_freq: Save checkpoint every N steps
        learning_rate: PPO learning rate
        n_steps: Steps per environment before update
        batch_size: Minibatch size
        n_epochs: Optimization epochs per update
        gamma: Discount factor
        device: 'cuda' or 'cpu'
    """

    # Create directories
    save_path = Path(save_dir)
    log_path = Path(log_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Starting PPO Training")
    print(f"  Parallel Environments: {n_envs}")
    print(f"  Total Timesteps: {total_timesteps:,}")
    print(f"  Device: {device}")
    print(f"  Save Directory: {save_path}")
    print(f"  Log Directory: {log_path}")

    # Create parallel environments using subprocess vectorization
    # This is faster than DummyVecEnv for multiple environments
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    # Create evaluation environment (single env)
    eval_env = DummyVecEnv([make_env(n_envs)])  # Use different namespace

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # Divide by n_envs because it counts steps per env
        save_path=str(save_path),
        name_prefix='ppo_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=True
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / 'best_model'),
        log_path=str(log_path),
        eval_freq=eval_freq // n_envs,
        deterministic=True,
        render=False,
        n_eval_episodes=5
    )

    # Create PPO agent
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,  # Entropy coefficient for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        stats_window_size=100,
        tensorboard_log=str(log_path),
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])  # 2-layer 256-unit policy and value networks
        ),
        verbose=1,
        device=device
    )

    print("\n📊 Model Architecture:")
    print(model.policy)

    # Train the agent
    print(f"\n🎓 Training started...")
    print(f"   Progress will be logged to TensorBoard: tensorboard --logdir {log_path}")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")

    # Save final model
    final_model_path = save_path / 'final_model'
    model.save(str(final_model_path))
    print(f"\n✅ Training complete! Final model saved to: {final_model_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return model


def main():
    parser = argparse.ArgumentParser(description='Train PPO agent for obstacle avoidance')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps')
    parser.add_argument('--n-envs', type=int, default=8, help='Number of parallel environments')
    parser.add_argument('--save-dir', type=str, default='models/ppo_navigation', help='Model save directory')
    parser.add_argument('--log-dir', type=str, default='logs/ppo_navigation', help='TensorBoard log directory')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Training device')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')

    args = parser.parse_args()

    # Initialize ROS
    rclpy.init()

    try:
        # Train
        model = train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_dir=args.save_dir,
            log_dir=args.log_dir,
            learning_rate=args.lr,
            device=args.device
        )
    finally:
        # Cleanup
        rclpy.shutdown()


if __name__ == '__main__':
    main()

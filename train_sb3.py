#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch

# Import our custom environment
from isaac_leatherback_env import LeatherbackEnv

def main():
    print("Starting Training with Stable Baselines 3...")
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")
    
    # Create Environment
    # Headless for faster training
    env = LeatherbackEnv(headless=True, use_camera=True)
    
    # Wrap in DummyVecEnv (SB3 requires this for performance/vectorization)
    env = DummyVecEnv([lambda: env])
    
    # Create PPO Model
    # MultiInputPolicy is needed because we have a Dict observation (Image + Vector)
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log="./leatherback_tensorboard/"
    )
    
    # Train
    total_timesteps = 100000
    print(f"Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps, progress_bar=True)
    
    # Save Model
    model.save("leatherback_ppo_v1")
    print("Model saved to leatherback_ppo_v1")
    
    # Close
    env.close()

if __name__ == "__main__":
    main()

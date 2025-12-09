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
    
    # Create the environment matching the Gym API
    env = LeatherbackEnv(headless=True, use_camera=True)
    
    # Wrap in DummyVecEnv (SB3 requires this for performance/vectorization)
    env = DummyVecEnv([lambda: env])
    
    # SENSOR VERIFICATION
    print("\n--- Sensor Check ---")
    obs = env.reset()
    # obs is a vectorized dict {key: array(batch_size, ...)}
    print(f"Keys: {obs.keys()}")
    
    img_data = obs['image'][0]
    lidar_data = obs['lidar'][0]
    vector_data = obs['vector'][0]
    
    print(f"Camera Shape: {img_data.shape}, Mean: {np.mean(img_data):.2f}")
    if np.mean(img_data) == 0:
        print("WARNING: Camera appears completely black/empty!")
    else:
        print("OK: Camera has data.")
        
    print(f"LiDAR Shape: {lidar_data.shape}, Range: [{np.min(lidar_data):.2f}, {np.max(lidar_data):.2f}]")
    if np.max(lidar_data) == 0:
         print("WARNING: LiDAR appears empty (all zeros)!")
    else:
         print("OK: LiDAR has data.")
         
    print("--------------------\n")
    
    # Check for PPO model file to load if retraining, OR create new
    # Since we changed physics/observation space, we MUST create NEW or reset
    # If we load old model, it will crash due to shape mismatch
    print("Creating NEW model for updated environment...")
    
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
    try:
        total_timesteps = 100000
        print(f"Training for {total_timesteps} steps...")
        model.learn(total_timesteps=total_timesteps, progress_bar=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by User! Saving current progress...")
    
    # Save Model
    model.save("leatherback_ppo_v1")
    print("Model saved to leatherback_ppo_v1")
    
    # Close
    env.close()

if __name__ == "__main__":
    main()

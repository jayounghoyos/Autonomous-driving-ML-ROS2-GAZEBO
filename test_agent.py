#!/usr/bin/env python3
"""
Test/Evaluation Script for Trained Leatherback Agent
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from isaac_leatherback_env import LeatherbackEnv
import torch

def main():
    print("="*60)
    print("Leatherback Agent Evaluation")
    print("="*60)
    
    # Check for saved model
    import os
    model_path = "leatherback_ppo_v1.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        print("Please run train_sb3.py first to generate the model.")
        return

    # Use CPU for inference usually fine, or CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Inference device: {device}")

    # Create Environment with GUI (Headless=False)
    # We want to SEE the robot now
    env = LeatherbackEnv(headless=False, use_camera=True)
    
    # Load Model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path, device=device)
    
    print("\nStarting evaluation loop...")
    print("Press Ctrl+C to stop.")
    
    obs, _ = env.reset()
    
    try:
        while env.sim_app.is_running():
            # Predict action
            # deterministic=True means pick best action, not random exploration
            action, _states = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print("Episode finished. Resetting...")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Evaluation stopped by user.")
        
    env.close()
    print("Evaluation complete.")

if __name__ == "__main__":
    main()

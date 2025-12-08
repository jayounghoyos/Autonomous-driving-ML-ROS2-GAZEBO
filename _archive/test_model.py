import gymnasium as gym
from stable_baselines3 import PPO
from vision_goal_env import VisionGoalEnv
import numpy as np
import time
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='interrupted_model', help='Path to model (zip)')
    args = parser.parse_args()

    # Verify model exists
    model_path = args.model
    if not model_path.endswith('.zip'):
        model_path += '.zip'
        
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return

    print(f"Loading Model: {model_path}")
    
    # Create Environment
    env = VisionGoalEnv()
    
    # Load Model
    model = PPO.load(model_path)
    
    print("Starting Test Drive...")
    obs, _ = env.reset()
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # Print status
            # vec_vals = [distance, angle, lin_vel, ang_vel]
            dist = obs['vector'][0]
            print(f"Dist: {dist:.2f}m | Reward: {reward:.2f} | Action: {action}")
            
            if terminated or truncated:
                print("Episode Finished (Collision or Timeout)")
                obs, _ = env.reset()
                time.sleep(1) # Pause before restart
                
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        env.close()

if __name__ == '__main__':
    main()

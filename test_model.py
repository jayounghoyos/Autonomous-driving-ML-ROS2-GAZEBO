#!/usr/bin/env python3
"""
Test the trained model - watch the robot navigate autonomously
"""
import rclpy
from goal_seeking_env import GoalSeekingEnv
from stable_baselines3 import PPO
import time

def main():
    print("Loading trained model...")
    rclpy.init()
    
    # Create environment
    env = GoalSeekingEnv()
    
    # Load trained model
    model = PPO.load('models/parallel_16env_final')
    print("Model loaded! Starting autonomous navigation...\n")
    
    # Run episodes
    for episode in range(10):
        obs, _ = env.reset()
        print(f"\nEpisode {episode + 1}: New goal at distance={obs[0]:.2f}m, angle={obs[1]:.2f}rad")
        
        done = False
        step = 0
        total_reward = 0
        
        while not done:
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            step += 1
            
            # Print progress every 20 steps
            if step % 20 == 0:
                print(f"  Step {step}: distance={obs[0]:.2f}m, reward={total_reward:.1f}")
            
            # Safety timeout
            if step > 500:
                print("  Timeout - episode too long")
                break
        
        if total_reward > 90:
            print(f"  SUCCESS! Goal reached in {step} steps, reward={total_reward:.1f}")
        else:
            print(f"  FAILED. Final distance={obs[0]:.2f}m, reward={total_reward:.1f}")
        
        time.sleep(2)  # Pause between episodes
    
    env.close()
    rclpy.shutdown()
    print("\nTest complete!")

if __name__ == '__main__':
    main()

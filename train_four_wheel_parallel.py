#!/usr/bin/env python3
"""
Parallel PPO Training for 4-Wheel Robot with Improved Rewards
Each environment runs its own isolated Gazebo instance
"""
import os
import subprocess
import time
import rclpy
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import sys
import argparse
sys.path.append(os.path.dirname(__file__))
from improved_goal_env import ImprovedGoalSeekingEnv


class IsolatedFourWheelEnv(ImprovedGoalSeekingEnv):
    """4-wheel environment that manages its own Gazebo instance"""
    
    def __init__(self, env_id):
        self.env_id = env_id
        self.gz_process = None
        self.bridge_process = None
        
        # Start isolated Gazebo server
        self._start_gazebo()
        time.sleep(3)
        
        # Start bridge
        self._start_bridge()
        time.sleep(1)
        
        # Initialize parent
        super().__init__()
        self.get_logger().info(f'4-Wheel Environment {env_id} ready!')
    
    def _start_gazebo(self):
        """Start headless Gazebo server"""
        env = os.environ.copy()
        env['GZ_SIM_SYSTEM_PLUGIN_PATH'] = '/opt/ros/kilted/opt/gz_sim_vendor/lib/gz-sim-9/plugins'
        env['GZ_PARTITION'] = f'env_{self.env_id}'
        
        cmd = [
            'gz', 'sim', '-s', '-r', '-v', '0',
            'src/vehicle_gazebo/worlds/rl_training_world.sdf'
        ]
        
        self.gz_process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def _start_bridge(self):
        """Start ROS-Gazebo bridge for 4 wheels"""
        env = os.environ.copy()
        env['GZ_PARTITION'] = f'env_{self.env_id}'
        
        cmd = [
            'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
            '/model/four_wheel_robot/joint/front_left_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/four_wheel_robot/joint/rear_left_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/four_wheel_robot/joint/front_right_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
            '/model/four_wheel_robot/joint/rear_right_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double',
            '/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry'
        ]
        
        self.bridge_process = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    
    def close(self):
        """Clean up processes"""
        super().close()
        if self.bridge_process:
            self.bridge_process.terminate()
        if self.gz_process:
            self.gz_process.terminate()


def make_env(rank):
    """Create isolated environment"""
    def _init():
        if not rclpy.ok():
            rclpy.init()
        return IsolatedFourWheelEnv(rank)
    return _init


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=8, 
                       help='Number of parallel environments (default: 8)')
    parser.add_argument('--timesteps', type=int, default=500000, 
                       help='Total training timesteps (default: 500k)')
    args = parser.parse_args()
    
    print(f"Starting Parallel PPO Training - 4-Wheel Robot")
    print(f"   Environments: {args.num_envs}")
    print(f"   Timesteps: {args.timesteps:,}")
    print(f"   Reward: Improved 7-component shaping")
    print(f"\nSystem Requirements:")
    print(f"   - RAM: ~{args.num_envs * 2}GB ({args.num_envs} envs × 2GB)")
    print(f"   - CPU: {args.num_envs} cores recommended")
    
    response = input(f"\nReady to launch {args.num_envs} Gazebo instances? (y/n): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print(f"\nCreating {args.num_envs} isolated environments...")
    print("   This will take ~30 seconds...")
    
    # Create parallel environments
    env = SubprocVecEnv([make_env(i) for i in range(args.num_envs)])
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // args.num_envs, 1000),
        save_path='./models/',
        name_prefix=f'four_wheel_{args.num_envs}env'
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
    print(f"  - Reward: Improved 7-component")
    print(f"  - Device: CUDA GPU")
    print(f"  - Parallel Environments: {args.num_envs}")
    print(f"  - Speedup: ~{args.num_envs}x faster!")
    print(f"  - Total Timesteps: {args.timesteps:,}")
    print(f"  - Checkpoints: Every {max(10000 // args.num_envs, 1000)} steps")
    print("\nTraining started! Press Ctrl+C to stop.\n")
    
    # Train
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True
        )
        
        model.save(f"./models/four_wheel_{args.num_envs}env_final")
        print(f"\nTraining complete! Model saved.")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted")
        model.save(f"./models/four_wheel_{args.num_envs}env_interrupted")
        print("Model saved.")
    
    finally:
        env.close()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Unified Training Script for Autonomous Driving
Supports Single (Debug) and Parallel (Speed) modes.
"""
import argparse
import os
import sys
import rclpy
import subprocess
import time
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

# Import Config and Environments
from config import TrainingConfig, RobotConfig
from vision_goal_env import VisionGoalEnv

class IsolatedVisionEnv(VisionGoalEnv):
    """
    Wraps the standard environment but manages its own
    isolated Gazebo + ROS instance.
    """
    def __init__(self, rank, is_gui=False):
        self.rank = rank
        self.processes = []
        
        # Unique Isolation IDs
        self.domain_id = 100 + rank
        self.gz_partition = f'env_{rank}'
        
        # Setup Envars
        self.env_vars = os.environ.copy()
        self.env_vars['ROS_DOMAIN_ID'] = str(self.domain_id)
        self.env_vars['GZ_PARTITION'] = self.gz_partition
        # Fix plugin path
        self.env_vars['GZ_SIM_SYSTEM_PLUGIN_PATH'] = '/opt/ros/kilted/opt/gz_sim_vendor/lib/gz-sim-9/plugins'

        # 1. Launch Gazebo
        self._launch_gazebo(gui=is_gui)
        time.sleep(5) # Wait for Gazebo
        
        # 2. Spawn Robot
        self._spawn_robot()
        time.sleep(2)
        
        # 3. Start Bridge
        self._start_bridge()
        time.sleep(1)
        
        # 4. Init Parent (VisionGoalEnv)
        # IMPORTANT: Parent needs to Init ROS with specific domain logic if not managed globally
        # But rclpy context is usually process-global. SubprocVecEnv forks, so we are safe.
        super().__init__()
        print(f'Parallel Env {rank} Ready!')

    def _launch_gazebo(self, gui):
        """Launch Gazebo Server (and Client if GUI)"""
        cmd = ['gz', 'sim', '-r', '-s', 'src/vehicle_gazebo/worlds/rl_training_world.sdf']
        if gui:
            # If GUI, run without -s (server only) usually, but 'gz sim' runs both
            cmd = ['gz', 'sim', '-r', 'src/vehicle_gazebo/worlds/rl_training_world.sdf']
        else:
             cmd = ['gz', 'sim', '-r', '-s', '-v', '0', 'src/vehicle_gazebo/worlds/rl_training_world.sdf']

        p = subprocess.Popen(cmd, env=self.env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(p)

    def _spawn_robot(self):
        """Spawn the robot model"""
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', 'rl_training_world',
            '-file', 'ackermann_rl_car.sdf',
            '-name', 'ackermann_rl_car',
            '-x', '0', '-y', '0', '-z', '0.18'
        ]
        p = subprocess.Popen(cmd, env=self.env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        p.wait() # Wait for spawn to finish

    def _start_bridge(self):
        """Start ROS-GZ Bridge"""
        # We need to bridge: Camera, Odom, Cmd_Vel, Goal Pose
        topics = [
            # Control
            f'/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist',
            # Sensors
            f'/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            f'/camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image',
            f'/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan',
            f'/model/goal_marker/pose@geometry_msgs/msg/Pose]gz.msgs.Pose'
        ]
        
        cmd = ['ros2', 'run', 'ros_gz_bridge', 'parameter_bridge'] + topics
        p = subprocess.Popen(cmd, env=self.env_vars, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.processes.append(p)

    def close(self):
        """Cleanup processes"""
        super().close()
        for p in self.processes:
            p.terminate()


def make_env(rank, is_parallel=True):
    """Factory"""
    def _init():
        if is_parallel:
            # Managed isolated environment
            return IsolatedVisionEnv(rank)
        else:
            # Standard attached environment (assumes external Gazebo)
            return VisionGoalEnv()
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='vision', help='Training mode')
    parser.add_argument('--parallel', type=int, default=0, help='Number of parallel environments (0=Single)')
    parser.add_argument('--timesteps', type=int, default=TrainingConfig.TOTAL_TIMESTEPS, help='Training Steps')
    args = parser.parse_args()

    print(f"🚀 Training Config: Parallel={args.parallel}, Steps={args.timesteps}")

    if args.parallel > 0:
        # Parallel Mode
        envs = [make_env(i, is_parallel=True) for i in range(args.parallel)]
        env = SubprocVecEnv(envs)
    else:
        # Single Mode (Debug)
        if not rclpy.ok(): rclpy.init()
        env = DummyVecEnv([make_env(0, is_parallel=False)])

    # Add Monitor
    env = VecMonitor(env)

    # Setup Checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // max(1, args.parallel), 1000),
        save_path='./models/',
        name_prefix=f'ppo_{args.parallel}env'
    )

    model = PPO(
        'MultiInputPolicy',
        env,
        verbose=1,
        device='cuda',
        learning_rate=TrainingConfig.LEARNING_RATE,
        n_steps=TrainingConfig.N_STEPS,
        batch_size=TrainingConfig.BATCH_SIZE,
        tensorboard_log='./tensorboard_logs/'
    )

    try:
        model.learn(total_timesteps=args.timesteps, callback=checkpoint_callback, progress_bar=True)
        model.save("final_model")
        print("✅ Done!")
    except KeyboardInterrupt:
        print("Interrupted.")
        model.save("interrupted_model")
    finally:
        env.close()
        if args.parallel == 0: rclpy.shutdown()

if __name__ == '__main__':
    main()

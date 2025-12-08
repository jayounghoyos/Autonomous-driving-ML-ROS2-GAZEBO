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
from raceway_goal_env import RacewayGoalEnv

class IsolatedVisionEnv(VisionGoalEnv):
    """
    Wraps the standard environment but manages its own
    isolated Gazebo + ROS instance.
    """
    def __init__(self, rank, is_gui=False, world='rl_training_world', env_class=VisionGoalEnv):
        self.rank = rank
        self.processes = []
        self.world_name = world
        self.env_class = env_class
        
        # Unique Isolation IDs
        self.domain_id = 100 + rank
        self.gz_partition = f'env_{rank}'
        
        # Setup Envars
        self.env_vars = os.environ.copy()
        self.env_vars['ROS_DOMAIN_ID'] = str(self.domain_id)
        self.env_vars['GZ_PARTITION'] = self.gz_partition
        
        # Initialize GZ_SIM_RESOURCE_PATH if not set
        if 'GZ_SIM_RESOURCE_PATH' not in self.env_vars:
            self.env_vars['GZ_SIM_RESOURCE_PATH'] = '/opt/ros/kilted/share'
        
        # Add urdf directory so robot SDF can be found and plugins load properly
        cwd = os.getcwd()
        self.env_vars['GZ_SIM_RESOURCE_PATH'] += f":{cwd}/src/vehicle_description/urdf"
        
        # Fix plugin path
        self.env_vars['GZ_SIM_SYSTEM_PLUGIN_PATH'] = '/opt/ros/kilted/opt/gz_sim_vendor/lib/gz-sim-9/plugins'

        # 1. Launch Gazebo
        self._launch_gazebo(gui=is_gui)
        time.sleep(5) # Wait for Gazebo
        
        # 2. Spawn Robot
        self._spawn_robot()
        time.sleep(5)  # Increased wait for Ackermann plugin to initialize
        
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
        if self.world_name == 'sonoma_raceway_training':
            sdf_file = 'sonoma_raceway_training.sdf'
            is_fuel = False
        elif self.world_name == 'warehouse_training':
            sdf_file = 'warehouse_training.sdf'
            is_fuel = False
        elif self.world_name == 'gymkhana_training':
            sdf_file = 'gymkhana_training.sdf'
            is_fuel = False
        elif self.world_name == 'depot_fuel':
            sdf_file = 'https://fuel.gazebosim.org/1.0/OpenRobotics/worlds/Depot'
            is_fuel = True
        elif self.world_name == 'harmonic_demo':
            # Use local cloned file
            sdf_file = 'src/harmonic_demo/harmonic_demo/harmonic.sdf'
            is_fuel = False
            # ADD RESOURCE PATH FOR HARMONIC MODELS
            # Point to the directory CONTAINING the models (Lake House, etc.)
            cwd = os.getcwd()
            self.env_vars['GZ_SIM_RESOURCE_PATH'] += f":{cwd}/src/harmonic_demo/harmonic_demo"

            is_fuel = False
        else:
            sdf_file = 'rl_training_world.sdf'
            is_fuel = False
        
        # Set resource paths BEFORE building command
        # This ensures models are found during world load
        if self.world_name in ['gymkhana_training', 'warehouse_training', 'sonoma_raceway_training']:
             cwd = os.getcwd()
             # Append models directory for local assets
             self.env_vars['GZ_SIM_RESOURCE_PATH'] += f":{cwd}/src/vehicle_gazebo/models"

        
        # Build command
        cmd = ['gz', 'sim']
        
        # Show GUI only for rank 0 (first environment)
        # This allows visual monitoring during parallel training
        if not gui and self.rank != 0:
            cmd.extend(['-s', '-v', '0']) # Headless for rank 1+
        # If gui=True OR rank=0, show the GUI
            
        # Add -r to run immediately
        cmd.append('-r')

        # Add file arg

        if is_fuel:
            cmd.append(sdf_file)
        elif self.world_name == 'harmonic_demo':
             # Absolute path for safety
             cmd.append(os.path.abspath(sdf_file))
        else:
            cmd.append(f'src/vehicle_gazebo/worlds/{sdf_file}')

        print(f"Launching Gazebo command: {' '.join(cmd)}", flush=True)
        
        # QT CONFLICT FIX: Sanitize environment for Gazebo
        # Remove ALL QT related variables to force system Qt usage
        gz_env = self.env_vars.copy()
        
        # DEBUG: Print resource path to verify it's set
        if 'GZ_SIM_RESOURCE_PATH' in gz_env:
            print(f"DEBUG: GZ_SIM_RESOURCE_PATH = {gz_env['GZ_SIM_RESOURCE_PATH']}", flush=True)
        else:
            print(f"WARNING: GZ_SIM_RESOURCE_PATH not set!", flush=True)
        
        keys_to_remove = [k for k in gz_env.keys() if k.startswith('QT_')]
        for k in keys_to_remove:
            print(f"Removing colliding env var: {k}={gz_env[k]}", flush=True)
            del gz_env[k]
            
        # Add local library path if needed
        # gz_env['LD_LIBRARY_PATH'] = ... 
            
        # Determine stdout/stderr destination
        stdout_dest = subprocess.DEVNULL
        stderr_dest = subprocess.DEVNULL
        if self.rank == 0: # For the first environment, show output for debugging
            stdout_dest = None
            stderr_dest = None

        p = subprocess.Popen(cmd, env=gz_env, stdout=stdout_dest, stderr=stderr_dest)
        self.processes.append(p)

    def _spawn_robot(self):
        """Spawn the robot model"""
        # Select spawn position based on world
        # Select spawn position based on world
        if self.world_name == 'sonoma_raceway_training':
            x, y, z = 0, 0, 0.3
            yaw = -0.663
        elif self.world_name == 'warehouse_training':
            x, y, z = -8, -8, 0.3
            yaw = 0.0
        elif self.world_name == 'gymkhana_training':
             x, y, z = 0, 0, 0.3
             yaw = 0.0
        elif self.world_name == 'depot_fuel':
             x, y, z = 0, 0, 0.5 # Slightly higher to be safe
             yaw = 0.0
        elif self.world_name == 'harmonic_demo':
             x, y, z = 0, 0, 0.5
             yaw = 0.0

        else:
            x, y, z = 0, 0, 0.18
            yaw = 0.0
            
        cmd = [
            'ros2', 'run', 'ros_gz_sim', 'create',
            '-world', self.world_name,
            '-file', 'ackermann_rl_car.sdf',  # Just filename - must be in GZ_SIM_RESOURCE_PATH!
            '-name', 'ackermann_rl_car',
            '-x', str(x), '-y', str(y), '-z', str(z), '-Y', str(yaw)
        ]
        # Enable stderr for rank 0 to see plugin errors
        stderr_dest = subprocess.DEVNULL if self.rank != 0 else None
        
        # CRITICAL: Run from project root where SDF can be found
        cwd = os.getcwd()
        print(f"DEBUG: Spawning from directory: {cwd}", flush=True)
        print(f"DEBUG: GZ_SIM_RESOURCE_PATH: {self.env_vars.get('GZ_SIM_RESOURCE_PATH')}", flush=True)
        
        p = subprocess.Popen(cmd, env=self.env_vars, cwd=cwd, stdout=subprocess.DEVNULL, stderr=stderr_dest)
        p.wait() # Wait for spawn to finish
        print(f"Robot spawn command completed with code: {p.returncode}", flush=True)

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


def make_env(rank, is_gui, env_type, world):
    """Factory for creating environments"""
    def _init():
        # Determine env_class and world based on env_type
        if env_type == 'raceway' or env_type == 'gymkhana' or env_type == 'depot' or env_type == 'harmonic':
            import raceway_goal_env
            env_cls = raceway_goal_env.RacewayGoalEnv
            
            if env_type == 'raceway': world_name_for_env = 'sonoma_raceway_training'
            elif env_type == 'gymkhana': world_name_for_env = 'gymkhana_training'
            elif env_type == 'depot': world_name_for_env = 'depot_fuel'
            else: world_name_for_env = 'harmonic_demo'
            
        elif env_type == 'warehouse':
            env_cls = VisionGoalEnv
            world_name_for_env = 'warehouse_training'
        elif env_type == 'vision':
            env_cls = VisionGoalEnv
            world_name_for_env = 'rl_training_world'
        else:
            raise ValueError(f"Unknown environment type: {env_type}")

        # Always use IsolatedVisionEnv to manage Gazebo, even for GUI
        # (Passes is_gui to IsolatedVisionEnv to decide on -s flag)
        return IsolatedVisionEnv(rank, is_gui=is_gui, world=world_name_for_env, env_class=env_cls)
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='vision', help='Training mode')
    parser.add_argument('--parallel', type=int, default=0, help='Number of parallel envs (0=single with GUI)')
    parser.add_argument('--gui', action='store_true', help='Show Gazebo GUI (Forces parallel=0)')
    parser.add_argument('--timesteps', type=int, default=500000, help='Total training timesteps')
    parser.add_argument('--env', type=str, default='vision', choices=['vision', 'raceway', 'warehouse', 'gymkhana', 'depot', 'harmonic'], help='Environment type')
    args = parser.parse_args()
    
    
    # Select world based on environment
    if args.env == 'raceway':
        world = 'sonoma_raceway_training'
    elif args.env == 'warehouse':
        world = 'warehouse_training'
    elif args.env == 'gymkhana':
        world = 'gymkhana_training'
    elif args.env == 'depot':
        world = 'depot_fuel'
    elif args.env == 'harmonic':
        world = 'harmonic_demo'

    else:
        world = 'rl_training_world'
    
    print(f"🚀 Training Config: Env={args.env}, Parallel={args.parallel}, Steps={args.timesteps}")
    # CREATE ENVIRONMENT
    if args.parallel > 0:
        # Parallel training
        envs = [make_env(i, False, args.env, world) for i in range(args.parallel)]
        env = SubprocVecEnv(envs)
    else:
        # Single environment (debug mode)
        if not rclpy.ok(): rclpy.init()
        env = DummyVecEnv([make_env(0, True, args.env, world)])

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
    
    print(f"🚀 Training hardware: {model.device}")
    print(f"ℹ️  If this says 'cpu', checking CUDA availability...")
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA is available: {torch.cuda.get_device_name(0)}")
    else:
        print(f"❌ CUDA NOT detected. Install torch with cuda support.")

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

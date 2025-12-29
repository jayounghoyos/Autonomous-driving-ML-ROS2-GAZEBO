#!/usr/bin/env python3
"""
Shadow Robot Controller - Isaac Sim + Real Robot.

Runs the trained model in Isaac Sim and publishes motor commands
to the real robot via ROS2.

The real robot "shadows" the simulated robot's movements.

PC (this script):
  - Runs Isaac Sim with virtual sensors
  - Runs trained RL policy
  - Publishes /cmd_vel to ROS2

Jetson Nano (motor_control_node.py):
  - Subscribes to /cmd_vel
  - Controls L298N motor driver
  - Real robot follows simulation

Usage:
  # On PC - start ROS2 daemon first
  ros2 daemon start

  # Run shadow controller
  $ISAAC_PYTHON training/shadow_robot.py --model models/ppo_jackal/diff_20251227_080037/final_model.zip

  # On Jetson Nano (separate terminal/SSH)
  python3 motor_control_node.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ROS2 imports (must be before Isaac Sim)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class ShadowPublisher(Node):
    """ROS2 publisher for shadow robot commands."""

    def __init__(self):
        super().__init__('shadow_robot_publisher')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("Shadow publisher initialized on /cmd_vel")

    def publish_cmd(self, linear: float, angular: float):
        """Publish velocity command."""
        msg = Twist()
        msg.linear.x = float(linear)
        msg.angular.z = float(angular)
        self.publisher.publish(msg)

    def stop(self):
        """Send stop command."""
        self.publish_cmd(0.0, 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shadow Robot Controller")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_differential_config.yaml",
        help="Config YAML",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Episodes to run")
    parser.add_argument("--scale-linear", type=float, default=1.0,
                        help="Scale factor for linear velocity (0.5 = half speed)")
    parser.add_argument("--scale-angular", type=float, default=1.0,
                        help="Scale factor for angular velocity")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between steps (seconds) for slower execution")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        model_path = Path(str(args.model) + ".zip")
        if not model_path.exists():
            print(f"Error: Model not found at {args.model}")
            return 1

    print("=" * 60)
    print("SHADOW ROBOT CONTROLLER")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print(f"Linear scale: {args.scale_linear}")
    print(f"Angular scale: {args.scale_angular}")
    print("=" * 60)

    # Initialize ROS2
    print("\nInitializing ROS2...")
    rclpy.init()
    shadow_pub = ShadowPublisher()

    # Import Isaac Sim modules (after ROS2 init)
    print("Importing Isaac Sim modules...")
    import yaml
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    project_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_root))

    from isaac_lab.envs import DifferentialDriveEnv, DifferentialDriveEnvCfg

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        robot_config = config.get("robot", {})
        env_config = config.get("env", {})
        print(f"Loaded config: {config_path}")
    else:
        robot_config = {}
        env_config = {}
        print("Using default config")

    # Get max velocities for scaling
    max_linear = env_config.get("max_linear_velocity", 2.0)
    max_angular = env_config.get("max_angular_velocity", 4.0)

    # Create environment WITH GUI
    print("\nCreating environment with GUI...")
    env_cfg = DifferentialDriveEnvCfg(
        robot_type=robot_config.get("type", "jackal"),
        wheel_radius=robot_config.get("wheel_radius", 0.098),
        track_width=robot_config.get("track_width", 0.37558),
        wheelbase=robot_config.get("wheelbase", 0.262),
        skid_steer_correction=robot_config.get("skid_steer_correction", 4.2),
        use_camera=env_config.get("use_camera", False),
        use_lidar=env_config.get("use_lidar", True),
        episode_length_s=env_config.get("episode_length_s", 120.0),
        goal_tolerance=env_config.get("goal_tolerance", 0.8),
        num_waypoints=env_config.get("num_waypoints", 3),
        waypoint_spacing=env_config.get("waypoint_spacing", 5.0),
        arena_radius=env_config.get("arena_radius", 25.0),
        lidar_num_points=env_config.get("lidar_num_points", 180),
        lidar_max_range=env_config.get("lidar_max_range", 20.0),
        num_obstacles_min=env_config.get("num_obstacles_min", 0),  # No obstacles for shadow test
        num_obstacles_max=env_config.get("num_obstacles_max", 0),
    )

    env = DifferentialDriveEnv(cfg=env_cfg, headless=False)

    # Load model
    print(f"\nLoading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), device=device)
    print(f"Model loaded (device: {device})")

    print("\n" + "=" * 60)
    print("SHADOW ROBOT ACTIVE")
    print("=" * 60)
    print("  Simulation: Isaac Sim (this window)")
    print("  Real robot: Following via ROS2 /cmd_vel")
    print("  Press Ctrl+C to stop")
    print("=" * 60)

    # Wait for GUI
    print("\nInitializing GUI (please wait)...")
    for _ in range(60):
        if not env.sim_app.is_running():
            break
        env._world.step(render=True)
        time.sleep(0.05)

    total_waypoints = 0
    episode = 0

    try:
        while env.sim_app.is_running() and episode < args.episodes:
            obs, info = env.reset()
            episode += 1
            episode_reward = 0.0
            step = 0
            waypoints_this_ep = 0

            print(f"\n{'='*40}")
            print(f"Episode {episode}/{args.episodes}")
            print(f"{'='*40}")

            done = False
            while not done and env.sim_app.is_running():
                # Get action from policy
                action, _ = model.predict(obs, deterministic=True)

                # Convert to velocity commands
                linear_vel = float(action[0]) * max_linear * args.scale_linear
                angular_vel = float(action[1]) * max_angular * args.scale_angular

                # Publish to real robot
                shadow_pub.publish_cmd(linear_vel, angular_vel)

                # Step simulation
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                done = terminated or truncated

                # Process ROS2 callbacks
                rclpy.spin_once(shadow_pub, timeout_sec=0)

                # Optional delay for slower execution
                if args.delay > 0:
                    time.sleep(args.delay)

                # Track waypoints
                if obs["vector"][0] < env.cfg.goal_tolerance:
                    waypoints_this_ep += 1

                # Status update
                if step % 100 == 0:
                    dist = obs["vector"][0]
                    print(f"  Step {step}: dist={dist:.1f}m, "
                          f"cmd=[{linear_vel:.2f}, {angular_vel:.2f}]")

            # Stop real robot at episode end
            shadow_pub.stop()

            total_waypoints += waypoints_this_ep
            print(f"  Episode ended: {waypoints_this_ep} waypoints, "
                  f"reward={episode_reward:.1f}")

            # Pause between episodes
            if episode < args.episodes and env.sim_app.is_running():
                print("\nNext episode in 3 seconds...")
                for _ in range(60):
                    if not env.sim_app.is_running():
                        break
                    env._world.step(render=True)
                    shadow_pub.stop()  # Keep robot stopped
                    rclpy.spin_once(shadow_pub, timeout_sec=0)
                    time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nStopping shadow robot...")

    finally:
        # Ensure real robot stops
        shadow_pub.stop()
        time.sleep(0.1)
        shadow_pub.stop()

        print("\n" + "=" * 60)
        print("Session Complete")
        print("=" * 60)
        print(f"  Episodes: {episode}")
        print(f"  Total waypoints: {total_waypoints}")
        print("=" * 60)

        shadow_pub.destroy_node()
        rclpy.shutdown()
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
View Trained Differential Drive Agent in Isaac Sim GUI.

Loads a trained model and runs it with visualization.

Usage:
    $ISAAC_PYTHON training/view_differential.py --model models/ppo_differential/.../final_model.zip
    $ISAAC_PYTHON training/view_differential.py --model models/ppo_differential/.../final_model.zip --stochastic
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View trained differential drive agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_differential_config.yaml",
        help="Config YAML",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to run")
    parser.add_argument("--slow", action="store_true", help="Slow motion")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic actions")
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
    print("Differential Drive Agent Viewer")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Config: {args.config}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Import after arg parsing
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

    # Create environment WITH GUI
    print("\nCreating environment with GUI...")
    env_cfg = DifferentialDriveEnvCfg(
        robot_type=robot_config.get("type", "jackal"),
        wheel_radius=robot_config.get("wheel_radius", 0.098),
        track_width=robot_config.get("track_width", 0.37558),
        wheelbase=robot_config.get("wheelbase", 0.262),
        skid_steer_correction=robot_config.get("skid_steer_correction", 4.2),
        use_camera=env_config.get("use_camera", True),
        use_lidar=env_config.get("use_lidar", True),
        episode_length_s=env_config.get("episode_length_s", 120.0),
        goal_tolerance=env_config.get("goal_tolerance", 0.5),
        num_waypoints=env_config.get("num_waypoints", 5),
        waypoint_spacing=env_config.get("waypoint_spacing", 4.0),
        arena_radius=env_config.get("arena_radius", 25.0),
        lidar_num_points=env_config.get("lidar_num_points", 180),
        lidar_max_range=env_config.get("lidar_max_range", 20.0),
        num_obstacles_min=env_config.get("num_obstacles_min", 3),
        num_obstacles_max=env_config.get("num_obstacles_max", 8),
    )

    env = DifferentialDriveEnv(cfg=env_cfg, headless=False)

    # Load model
    print(f"\nLoading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), device=device)
    print(f"Model loaded (device: {device})")

    # Action mode
    deterministic = not args.stochastic
    action_mode = "deterministic" if deterministic else "stochastic (exploration)"

    print("\n" + "=" * 60)
    print("Starting visualization - Close Isaac Sim window to exit")
    print(f"Action mode: {action_mode}")
    print("Press Ctrl+C to stop early")
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
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                done = terminated or truncated

                # Track waypoints
                if obs["vector"][0] < env.cfg.goal_tolerance:
                    waypoints_this_ep += 1

                if args.slow:
                    time.sleep(0.02)

                if step % 200 == 0:
                    dist = obs["vector"][0]
                    print(f"  Step {step}: dist={dist:.1f}m, reward={episode_reward:.1f}")

            total_waypoints += waypoints_this_ep

            if waypoints_this_ep >= env.cfg.num_waypoints:
                print(f"  SUCCESS! All {waypoints_this_ep} waypoints reached!")
            elif terminated:
                print(f"  Terminated after {waypoints_this_ep} waypoints")
            else:
                print(f"  Truncated after {waypoints_this_ep} waypoints")

            print(f"  Total reward: {episode_reward:.2f}")

            # Pause between episodes
            if episode < args.episodes and env.sim_app.is_running():
                print("\nNext episode in 3 seconds...")
                for _ in range(60):
                    if not env.sim_app.is_running():
                        break
                    env._world.step(render=True)
                    time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        print("\n" + "=" * 60)
        print("Session Complete")
        print("=" * 60)
        print(f"  Episodes: {episode}")
        print(f"  Total waypoints: {total_waypoints}")
        print("=" * 60)
        env.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())

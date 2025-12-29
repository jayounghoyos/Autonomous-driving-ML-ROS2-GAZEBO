#!/usr/bin/env python3
"""
Evaluation Script for Leatherback Navigation.

This script evaluates a trained PPO agent on the Leatherback vehicle
navigation task using Isaac Sim 5.1.0.

Usage:
    python training/evaluate.py --model models/ppo/final_model.zip
    python training/evaluate.py --model models/ppo/final_model.zip --episodes 10
    python training/evaluate.py --model models/ppo/final_model.zip --render
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained Leatherback agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained model (.zip file)",
    )

    # Evaluation settings
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=2000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions (overrides --deterministic)",
    )

    # Environment
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable RGB camera",
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="Enable LiDAR",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_config.yaml",
        help="Path to config YAML (to match training settings)",
    )

    # Output
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print detailed step information",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save video of evaluation (requires camera)",
    )

    return parser.parse_args()


def main() -> int:
    """Main evaluation function."""
    args = parse_args()

    # Check model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return 1

    # Determine deterministic
    deterministic = args.deterministic and not args.stochastic

    print("=" * 70)
    print("Leatherback Evaluation")
    print("=" * 70)
    print(f"  Model: {model_path}")
    print(f"  Config: {args.config}")
    print(f"  Episodes: {args.episodes}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Deterministic: {deterministic}")
    print(f"  Headless: {args.headless}")
    print("=" * 70)

    # Import after parsing (allows --help without Isaac Sim)
    print("\nInitializing Isaac Sim...")
    import torch
    import yaml
    from stable_baselines3 import PPO

    # Add project to path
    project_root = Path(__file__).parent.parent.absolute()
    sys.path.insert(0, str(project_root))

    from isaac_lab.envs import LeatherbackEnv, LeatherbackEnvCfg

    # Load config to match training settings
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        env_config = config.get("env", {})
        print(f"  Loaded config from: {config_path}")
    else:
        env_config = {}
        print(f"  Warning: Config not found, using defaults")

    # Create environment with same settings as training
    print("\nCreating environment...")
    env_cfg = LeatherbackEnvCfg(
        use_camera=args.camera or env_config.get("use_camera", False),
        use_lidar=args.lidar or env_config.get("use_lidar", False),
        episode_length_s=env_config.get("episode_length_s", 60.0),
        goal_tolerance=env_config.get("goal_tolerance", 0.5),
        num_waypoints=env_config.get("num_waypoints", 5),
        waypoint_spacing=env_config.get("waypoint_spacing", 4.0),
        arena_radius=env_config.get("arena_radius", 25.0),
        # Obstacle settings
        num_obstacles_min=env_config.get("num_obstacles_min", 3),
        num_obstacles_max=env_config.get("num_obstacles_max", 8),
        obstacle_spawn_radius_min=env_config.get("obstacle_spawn_radius_min", 6.0),
        obstacle_spawn_radius_max=env_config.get("obstacle_spawn_radius_max", 18.0),
    )
    env = LeatherbackEnv(cfg=env_cfg, headless=args.headless)

    # Load model
    print(f"\nLoading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), device=device)
    print(f"Model loaded (device: {device})")

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    waypoints_reached = []
    success_count = 0

    print("\n" + "=" * 70)
    print("Starting evaluation...")
    print("=" * 70)

    try:
        for episode in range(args.episodes):
            obs, info = env.reset()
            episode_reward = 0.0
            step = 0
            waypoints = 0
            done = False

            print(f"\nEpisode {episode + 1}/{args.episodes}")

            while not done and step < args.max_steps:
                # Check if simulation is still running
                if not env.sim_app.is_running():
                    print("Simulation closed by user")
                    break

                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step += 1
                done = terminated or truncated

                # Track waypoints (check distance threshold)
                distance = obs["vector"][0]
                if distance < env.cfg.goal_tolerance:
                    waypoints += 1

                # Verbose output
                if args.verbose and step % 50 == 0:
                    print(
                        f"  Step {step}: dist={distance:.2f}m, "
                        f"reward={reward:.2f}, total={episode_reward:.2f}"
                    )

            # Episode complete
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            waypoints_reached.append(waypoints)

            # Check success (reached all waypoints)
            if waypoints >= env.cfg.num_waypoints:
                success_count += 1
                status = "SUCCESS"
            elif terminated:
                status = "TERMINATED"
            else:
                status = "TRUNCATED"

            print(
                f"  Result: {status} | "
                f"Steps: {step} | "
                f"Waypoints: {waypoints}/{env.cfg.num_waypoints} | "
                f"Reward: {episode_reward:.2f}"
            )

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")

    finally:
        env.close()

    # Print summary statistics
    if episode_rewards:
        print("\n" + "=" * 70)
        print("Evaluation Summary")
        print("=" * 70)
        print(f"  Episodes completed: {len(episode_rewards)}")
        print(f"  Success rate: {success_count}/{len(episode_rewards)} "
              f"({100 * success_count / len(episode_rewards):.1f}%)")
        print(f"\n  Reward:")
        print(f"    Mean: {np.mean(episode_rewards):.2f}")
        print(f"    Std:  {np.std(episode_rewards):.2f}")
        print(f"    Min:  {np.min(episode_rewards):.2f}")
        print(f"    Max:  {np.max(episode_rewards):.2f}")
        print(f"\n  Episode length:")
        print(f"    Mean: {np.mean(episode_lengths):.1f}")
        print(f"    Std:  {np.std(episode_lengths):.1f}")
        print(f"\n  Waypoints reached:")
        print(f"    Mean: {np.mean(waypoints_reached):.1f}/{env.cfg.num_waypoints}")
        print(f"    Max:  {np.max(waypoints_reached)}/{env.cfg.num_waypoints}")
        print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

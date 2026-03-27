#!/usr/bin/env python3
"""
PPO Training Script for Leatherback Navigation.

This script trains a PPO agent to navigate the Leatherback vehicle
through waypoints using Isaac Sim 5.1.0 + Stable-Baselines3.

Usage:
    python training/train_ppo.py --headless
    python training/train_ppo.py --config training/configs/ppo_config.yaml
    python training/train_ppo.py --resume models/ppo/checkpoint_50000.zip
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Leatherback navigation with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_config.yaml",
        help="Path to training configuration YAML",
    )

    # Training
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps from config",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
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

    # Output
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (for logging)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration")
        return {}

    with open(config_file) as f:
        return yaml.safe_load(f)


def main() -> int:
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    ppo_config = config.get("ppo", {})
    training_config = config.get("training", {})
    paths_config = config.get("paths", {})
    hardware_config = config.get("hardware", {})
    env_config = config.get("env", {})

    # Get project root
    project_root = Path(__file__).parent.parent.absolute()

    # Override with command line arguments
    total_timesteps = args.timesteps or training_config.get("total_timesteps", 500000)
    use_camera = args.camera or env_config.get("use_camera", False)
    use_lidar = args.lidar or env_config.get("use_lidar", False)
    seed = args.seed or hardware_config.get("seed")

    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"ppo_{timestamp}"

    save_dir = project_root / paths_config.get("save_dir", "models/ppo") / exp_name
    log_dir = project_root / paths_config.get("log_dir", "logs/ppo")
    tensorboard_dir = project_root / paths_config.get("tensorboard_dir", "logs/tensorboard")

    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_tensorboard:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Leatherback PPO Training")
    print("=" * 70)
    print(f"  Config: {args.config}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Headless: {args.headless}")
    print(f"  Sensors:")
    print(f"    Camera: {use_camera}" + (f" ({env_config.get('camera_resolution', [64,64])})" if use_camera else ""))
    print(f"    LiDAR:  {use_lidar}" + (f" ({env_config.get('lidar_num_points', 180)} points)" if use_lidar else ""))
    print(f"    Vector: True (distance, heading, prev_action)")
    print(f"  Obstacles: {env_config.get('num_obstacles_min', 3)}-{env_config.get('num_obstacles_max', 8)} (randomized)")
    print(f"  Save dir: {save_dir}")
    if args.resume:
        print(f"  Resume from: {args.resume}")
    print("=" * 70)

    # Import after parsing (allows --help without Isaac Sim)
    print("\nInitializing Isaac Sim...")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    sys.path.insert(0, str(project_root))
    from training.callbacks import linear_schedule, TrainingMetricsCallback

    from isaac_lab.envs import LeatherbackEnv, LeatherbackEnvCfg

    # Device selection
    device_setting = hardware_config.get("device", "auto")
    if device_setting == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_setting

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create environment configuration
    env_cfg = LeatherbackEnvCfg(
        use_camera=use_camera,
        use_lidar=use_lidar,
        episode_length_s=env_config.get("episode_length_s", 60.0),
        goal_tolerance=env_config.get("goal_tolerance", 0.5),
        num_waypoints=env_config.get("num_waypoints", 10),
        waypoint_spacing=env_config.get("waypoint_spacing", 5.0),
        arena_radius=env_config.get("arena_radius", 12.0),
        # Camera settings
        camera_resolution=tuple(env_config.get("camera_resolution", [64, 64])),
        camera_position=tuple(env_config.get("camera_position", [0.8, 0.0, 0.8])),
        # LiDAR settings
        lidar_num_points=env_config.get("lidar_num_points", 180),
        lidar_max_range=env_config.get("lidar_max_range", 20.0),
        lidar_position=tuple(env_config.get("lidar_position", [0.0, 0.0, 1.0])),
        # Obstacle settings
        num_obstacles_min=env_config.get("num_obstacles_min", 3),
        num_obstacles_max=env_config.get("num_obstacles_max", 8),
        obstacle_spawn_radius_min=env_config.get("obstacle_spawn_radius_min", 3.0),
        obstacle_spawn_radius_max=env_config.get("obstacle_spawn_radius_max", 15.0),
        obstacle_size_min=tuple(env_config.get("obstacle_size_min", [0.5, 0.5, 0.5])),
        obstacle_size_max=tuple(env_config.get("obstacle_size_max", [2.0, 2.0, 1.5])),
        randomize_obstacle_colors=env_config.get("randomize_obstacle_colors", True),
        randomize_obstacles_on_reset=env_config.get("randomize_obstacles_on_reset", True),
        obstacle_min_spawn_distance=env_config.get("obstacle_min_spawn_distance", 2.0),
    )

    # Create environment
    print("\nCreating environment...")
    env = LeatherbackEnv(cfg=env_cfg, headless=args.headless)
    vec_env = DummyVecEnv([lambda: env])

    # LR schedule and VecNormalize path
    vecnorm_path = save_dir / "vecnormalize.pkl"
    use_lr_schedule = ppo_config.get("use_lr_schedule", True)
    learning_rate_value = ppo_config.get("learning_rate", 3e-4)
    learning_rate = linear_schedule(learning_rate_value) if use_lr_schedule else learning_rate_value

    # Create or load model
    if args.resume:
        print(f"\nLoading model from {args.resume}...")
        resume_path = Path(args.resume)
        resume_vecnorm = resume_path.parent / "vecnormalize.pkl"
        if resume_vecnorm.exists():
            vec_env = VecNormalize.load(str(resume_vecnorm), vec_env)
            vec_env.training = True
            vec_env.norm_reward = True
            print(f"  VecNormalize stats loaded from {resume_vecnorm}")
        else:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
            print("  No VecNormalize stats found, starting fresh normalization")

        model = PPO.load(
            args.resume,
            env=vec_env,
            device=device,
            tensorboard_log=None if args.no_tensorboard else str(tensorboard_dir),
        )
        print("Model loaded successfully")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

        print("\nCreating new PPO model...")
        model = PPO(
            policy=ppo_config.get("policy", "MlpPolicy"),
            env=vec_env,
            learning_rate=learning_rate,
            n_steps=ppo_config.get("n_steps", 2048),
            batch_size=ppo_config.get("batch_size", 64),
            n_epochs=ppo_config.get("n_epochs", 10),
            gamma=ppo_config.get("gamma", 0.99),
            gae_lambda=ppo_config.get("gae_lambda", 0.95),
            clip_range=ppo_config.get("clip_range", 0.2),
            ent_coef=ppo_config.get("ent_coef", 0.01),
            vf_coef=ppo_config.get("vf_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            verbose=1,
            device=device,
            seed=seed,
            tensorboard_log=None if args.no_tensorboard else str(tensorboard_dir),
        )
        print("Model created")

    # Setup callbacks
    callbacks = []

    # Checkpoint — saves vecnormalize_{step}.pkl alongside model checkpoints
    checkpoint_freq = training_config.get("checkpoint_freq", 50000)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Metrics callback — tracks success_rate in TensorBoard without a second env
    metrics_callback = TrainingMetricsCallback(
        window=training_config.get("metrics_window", 50),
        log_freq=ppo_config.get("n_steps", 2048),
    )
    callbacks.append(metrics_callback)

    callback_list = CallbackList(callbacks)

    # Save config for reproducibility
    config_save_path = save_dir / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {config_save_path}")

    # Training
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70)

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback_list,
            log_interval=training_config.get("log_interval", 10),
            progress_bar=True,
            reset_num_timesteps=args.resume is None,
        )
        print("\nTraining completed successfully!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Save final model
        final_model_path = save_dir / "final_model"
        model.save(str(final_model_path))
        print(f"\nFinal model saved to {final_model_path}")

        # Save VecNormalize stats — required to run the model correctly at inference
        vec_env.save(str(vecnorm_path))
        print(f"VecNormalize stats saved to {vecnorm_path}")

        vec_env.close()
        print("Environment closed")

    print("\n" + "=" * 70)
    print("Training session complete")
    print(f"Models saved in: {save_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

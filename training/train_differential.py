#!/usr/bin/env python3
"""
PPO Training Script for 4-Wheel Differential Drive Robot.

Trains a PPO agent to navigate a skid-steer robot through waypoints
using Isaac Sim 5.1.0 + Stable-Baselines3.

This robot matches typical hobby hardware:
- 4 DC motors (2 per side)
- L298N motor driver
- Jetson Nano compute

Usage:
    $ISAAC_PYTHON training/train_differential.py --headless
    $ISAAC_PYTHON training/train_differential.py --config training/configs/ppo_differential_config.yaml
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
        description="Train Differential Drive robot with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_differential_config.yaml",
        help="Training config YAML",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total timesteps",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI",
    )
    parser.add_argument(
        "--camera",
        action="store_true",
        help="Enable camera",
    )
    parser.add_argument(
        "--lidar",
        action="store_true",
        help="Enable LiDAR",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard",
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config not found at {config_path}, using defaults")
        return {}

    with open(config_file) as f:
        return yaml.safe_load(f)


def main() -> int:
    """Main training function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)
    robot_config = config.get("robot", {})
    env_config = config.get("env", {})
    ppo_config = config.get("ppo", {})
    training_config = config.get("training", {})
    paths_config = config.get("paths", {})
    hardware_config = config.get("hardware", {})

    # Get robot type
    robot_type = robot_config.get("type", "jackal")

    project_root = Path(__file__).parent.parent.absolute()

    # Override with CLI args
    total_timesteps = args.timesteps or training_config.get("total_timesteps", 500000)
    use_camera = args.camera or env_config.get("use_camera", False)
    use_lidar = args.lidar or env_config.get("use_lidar", False)
    seed = args.seed or hardware_config.get("seed")

    # Setup paths
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"diff_{timestamp}"

    save_dir = project_root / paths_config.get("save_dir", "models/ppo_differential") / exp_name
    log_dir = project_root / paths_config.get("log_dir", "logs/ppo_differential")
    tensorboard_dir = project_root / paths_config.get("tensorboard_dir", "logs/tensorboard")

    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_tensorboard:
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Clearpath {robot_type.upper()} - PPO Training")
    print("=" * 70)
    print(f"  Config: {args.config}")
    print(f"  Robot: {robot_type}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Headless: {args.headless}")
    print(f"  Robot Geometry:")
    print(f"    Wheel radius: {robot_config.get('wheel_radius', 0.098)}m")
    print(f"    Track width:  {robot_config.get('track_width', 0.37558)}m")
    print(f"    Wheelbase:    {robot_config.get('wheelbase', 0.262)}m")
    print(f"  Sensors:")
    print(f"    Camera: {use_camera}" + (f" ({env_config.get('camera_resolution', [64,64])})" if use_camera else ""))
    print(f"    LiDAR:  {use_lidar}" + (f" ({env_config.get('lidar_num_points', 180)} points)" if use_lidar else ""))
    print(f"    Vector: True (distance, heading, prev_action)")
    print(f"  Obstacle Course:")
    print(f"    Type: {env_config.get('obstacle_course_type', 'barn')}")
    print(f"    Shape: {env_config.get('obstacle_shape', 'cylinder')}")
    print(f"    Count: {env_config.get('num_obstacles_min', 5)}-{env_config.get('num_obstacles_max', 40)}")
    if env_config.get('use_curriculum', True):
        print(f"    Curriculum: {env_config.get('curriculum_start_difficulty', 0.2)} -> {env_config.get('curriculum_end_difficulty', 0.9)}")
    else:
        print(f"    Difficulty: {env_config.get('obstacle_difficulty', 0.5)}")
    print(f"  Save dir: {save_dir}")
    if args.resume:
        print(f"  Resume: {args.resume}")
    print("=" * 70)

    # Import after arg parsing
    print("\nInitializing Isaac Sim...")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    sys.path.insert(0, str(project_root))

    from isaac_lab.envs import DifferentialDriveEnv, DifferentialDriveEnvCfg

    # Device
    device_setting = hardware_config.get("device", "auto")
    if device_setting == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = device_setting

    print(f"Using device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Create environment config
    env_cfg = DifferentialDriveEnvCfg(
        # Robot selection
        robot_type=robot_type,
        # Robot geometry (from config or use defaults for robot type)
        wheel_radius=robot_config.get("wheel_radius", 0.098),
        track_width=robot_config.get("track_width", 0.37558),
        wheelbase=robot_config.get("wheelbase", 0.262),
        robot_mass=robot_config.get("robot_mass", 17.0),
        skid_steer_correction=robot_config.get("skid_steer_correction", 4.2),
        # Sensors
        use_camera=use_camera,
        use_lidar=use_lidar,
        camera_resolution=tuple(env_config.get("camera_resolution", [64, 64])),
        camera_position=tuple(env_config.get("camera_position", [0.15, 0.0, 0.15])),
        lidar_num_points=env_config.get("lidar_num_points", 180),
        lidar_max_range=env_config.get("lidar_max_range", 10.0),
        lidar_position=tuple(env_config.get("lidar_position", [0.0, 0.0, 0.20])),
        # Navigation
        episode_length_s=env_config.get("episode_length_s", 120.0),
        goal_tolerance=env_config.get("goal_tolerance", 0.5),
        num_waypoints=env_config.get("num_waypoints", 5),
        waypoint_spacing=env_config.get("waypoint_spacing", 4.0),
        waypoint_lateral_range=env_config.get("waypoint_lateral_range", 3.0),
        arena_radius=env_config.get("arena_radius", 25.0),
        # Progressive goal curriculum (v3)
        use_progressive_goals=env_config.get("use_progressive_goals", False),
        stage1_episodes=env_config.get("stage1_episodes", 200),
        stage1_num_waypoints=env_config.get("stage1_num_waypoints", 1),
        stage1_goal_distance_min=env_config.get("stage1_goal_distance_min", 4.0),
        stage1_goal_distance_max=env_config.get("stage1_goal_distance_max", 6.0),
        stage1_lateral_range=env_config.get("stage1_lateral_range", 1.0),
        stage2_episodes=env_config.get("stage2_episodes", 400),
        stage2_num_waypoints=env_config.get("stage2_num_waypoints", 2),
        stage2_goal_spacing=env_config.get("stage2_goal_spacing", 5.0),
        stage2_lateral_range=env_config.get("stage2_lateral_range", 1.5),
        stage3_episodes=env_config.get("stage3_episodes", 600),
        stage3_num_waypoints=env_config.get("stage3_num_waypoints", 3),
        stage3_goal_spacing=env_config.get("stage3_goal_spacing", 5.0),
        stage3_lateral_range=env_config.get("stage3_lateral_range", 2.0),
        stage4_num_waypoints=env_config.get("stage4_num_waypoints", 3),
        stage4_goal_spacing=env_config.get("stage4_goal_spacing", 5.0),
        stage4_lateral_range=env_config.get("stage4_lateral_range", 2.5),
        # Motor limits
        max_wheel_velocity=env_config.get("max_wheel_velocity", 12.0),
        max_linear_velocity=env_config.get("max_linear_velocity", 0.8),
        max_angular_velocity=env_config.get("max_angular_velocity", 2.0),
        # BARN-style obstacle course
        obstacle_course_type=env_config.get("obstacle_course_type", "barn"),
        obstacle_shape=env_config.get("obstacle_shape", "cylinder"),
        obstacle_difficulty=env_config.get("obstacle_difficulty", 0.5),
        use_curriculum=env_config.get("use_curriculum", True),
        curriculum_start_difficulty=env_config.get("curriculum_start_difficulty", 0.2),
        curriculum_end_difficulty=env_config.get("curriculum_end_difficulty", 0.9),
        curriculum_episodes_to_max=env_config.get("curriculum_episodes_to_max", 500),
        # Obstacle counts
        num_obstacles_min=env_config.get("num_obstacles_min", 5),
        num_obstacles_max=env_config.get("num_obstacles_max", 40),
        # Cylinder dimensions
        cylinder_radius_min=env_config.get("cylinder_radius_min", 0.15),
        cylinder_radius_max=env_config.get("cylinder_radius_max", 0.4),
        cylinder_height_min=env_config.get("cylinder_height_min", 0.5),
        cylinder_height_max=env_config.get("cylinder_height_max", 1.2),
        # Cube dimensions (for mixed mode)
        obstacle_size_min=tuple(env_config.get("obstacle_size_min", [0.3, 0.3, 0.5])),
        obstacle_size_max=tuple(env_config.get("obstacle_size_max", [0.8, 0.8, 1.0])),
        # Course dimensions
        course_width=env_config.get("course_width", 8.0),
        course_length=env_config.get("course_length", 20.0),
        obstacle_spawn_x_min=env_config.get("obstacle_spawn_x_min", 2.0),
        obstacle_spawn_x_max=env_config.get("obstacle_spawn_x_max", 18.0),
        # Passage width
        min_passage_width=env_config.get("min_passage_width", 0.8),
        max_passage_width=env_config.get("max_passage_width", 2.0),
        # Safety clearances
        obstacle_min_spawn_distance=env_config.get("obstacle_min_spawn_distance", 1.5),
        obstacle_waypoint_clearance=env_config.get("obstacle_waypoint_clearance", 1.0),
        randomize_obstacle_colors=env_config.get("randomize_obstacle_colors", True),
        randomize_obstacles_on_reset=env_config.get("randomize_obstacles_on_reset", True),
        # Reward configuration
        reward_progress_scale=env_config.get("reward_progress_scale", 30.0),
        reward_away_penalty_scale=env_config.get("reward_away_penalty_scale", 15.0),
        reward_heading_scale=env_config.get("reward_heading_scale", 0.5),
        reward_velocity_scale=env_config.get("reward_velocity_scale", 0.3),
        reward_smooth_scale=env_config.get("reward_smooth_scale", 0.1),
        reward_reverse_penalty=env_config.get("reward_reverse_penalty", -0.3),
        reward_waypoint_bonus=env_config.get("reward_waypoint_bonus", 200.0),
        reward_all_waypoints_bonus=env_config.get("reward_all_waypoints_bonus", 500.0),
        reward_collision_penalty=env_config.get("reward_collision_penalty", -150.0),
        reward_obstacle_danger_zone=env_config.get("reward_obstacle_danger_zone", 2.0),
        reward_obstacle_penalty_max=env_config.get("reward_obstacle_penalty_max", 8.0),
        reward_stuck_penalty=env_config.get("reward_stuck_penalty", -2.0),
        stuck_threshold_steps=env_config.get("stuck_threshold_steps", 30),
        stuck_movement_threshold=env_config.get("stuck_movement_threshold", 0.05),
    )

    # Create environment
    print("\nCreating environment...")
    env = DifferentialDriveEnv(cfg=env_cfg, headless=args.headless)

    # Wrap for SB3
    vec_env = DummyVecEnv([lambda: env])

    # Create or load model
    if args.resume:
        print(f"\nLoading model from {args.resume}...")
        model = PPO.load(
            args.resume,
            env=vec_env,
            device=device,
            tensorboard_log=None if args.no_tensorboard else str(tensorboard_dir),
        )
        print("Model loaded")
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            policy=ppo_config.get("policy", "MlpPolicy"),
            env=vec_env,
            learning_rate=ppo_config.get("learning_rate", 3e-4),
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

    # Callbacks
    callbacks = []
    checkpoint_freq = training_config.get("checkpoint_freq", 50000)
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_dir),
        name_prefix="checkpoint",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)
    callback_list = CallbackList(callbacks)

    # Save config
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
        print("\nTraining completed!")

    except KeyboardInterrupt:
        print("\n\nTraining interrupted")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Save final model
        final_model_path = save_dir / "final_model"
        model.save(str(final_model_path))
        print(f"\nFinal model saved to {final_model_path}")

        env.close()
        print("Environment closed")

    print("\n" + "=" * 70)
    print("Training session complete")
    print(f"Models saved in: {save_dir}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

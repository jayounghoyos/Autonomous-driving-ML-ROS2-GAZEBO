#!/usr/bin/env python3
"""
Preview BARN-Style Obstacle Course.

Shows the obstacle course without running any agent.
Use this to visualize the environment before training.

Usage:
    $ISAAC_PYTHON training/preview_barn.py
    $ISAAC_PYTHON training/preview_barn.py --difficulty 0.8
    $ISAAC_PYTHON training/preview_barn.py --course-type corridor
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview BARN obstacle course")
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/ppo_barn_config.yaml",
        help="Config YAML",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=None,
        help="Override difficulty (0.0-1.0)",
    )
    parser.add_argument(
        "--course-type",
        type=str,
        choices=["barn", "corridor", "maze", "random"],
        default=None,
        help="Override course type",
    )
    parser.add_argument(
        "--num-obstacles",
        type=int,
        default=None,
        help="Override number of obstacles",
    )
    parser.add_argument(
        "--resets",
        type=int,
        default=5,
        help="Number of resets to preview",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("=" * 60)
    print("BARN Obstacle Course Preview")
    print("=" * 60)

    # Import after arg parsing
    import yaml
    import numpy as np

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
        print(f"Config: {config_path}")
    else:
        robot_config = {}
        env_config = {}
        print("Using defaults")

    # Apply overrides
    if args.difficulty is not None:
        env_config["obstacle_difficulty"] = args.difficulty
        env_config["use_curriculum"] = False  # Disable curriculum for fixed difficulty
    if args.course_type is not None:
        env_config["obstacle_course_type"] = args.course_type
    if args.num_obstacles is not None:
        env_config["num_obstacles_min"] = args.num_obstacles
        env_config["num_obstacles_max"] = args.num_obstacles

    # Print settings
    course_type = env_config.get("obstacle_course_type", "barn")
    difficulty = env_config.get("obstacle_difficulty", 0.5)
    use_curriculum = env_config.get("use_curriculum", True)
    num_min = env_config.get("num_obstacles_min", 8)
    num_max = env_config.get("num_obstacles_max", 30)

    print(f"Course type: {course_type}")
    print(f"Obstacle shape: {env_config.get('obstacle_shape', 'cylinder')}")
    print(f"Obstacles: {num_min}-{num_max}")
    if use_curriculum:
        print(f"Curriculum: {env_config.get('curriculum_start_difficulty', 0.2)} -> {env_config.get('curriculum_end_difficulty', 0.8)}")
    else:
        print(f"Fixed difficulty: {difficulty}")
    print("=" * 60)

    # Create environment WITH GUI
    print("\nCreating environment with GUI...")
    env_cfg = DifferentialDriveEnvCfg(
        robot_type=robot_config.get("type", "jackal"),
        wheel_radius=robot_config.get("wheel_radius", 0.098),
        track_width=robot_config.get("track_width", 0.37558),
        wheelbase=robot_config.get("wheelbase", 0.262),
        skid_steer_correction=robot_config.get("skid_steer_correction", 4.2),
        use_camera=False,
        use_lidar=env_config.get("use_lidar", True),
        episode_length_s=env_config.get("episode_length_s", 90.0),
        goal_tolerance=env_config.get("goal_tolerance", 0.8),
        num_waypoints=env_config.get("num_waypoints", 3),
        waypoint_spacing=env_config.get("waypoint_spacing", 5.0),
        waypoint_lateral_range=env_config.get("waypoint_lateral_range", 2.0),
        arena_radius=env_config.get("arena_radius", 25.0),
        lidar_num_points=env_config.get("lidar_num_points", 180),
        lidar_max_range=env_config.get("lidar_max_range", 10.0),
        # BARN obstacle course settings
        obstacle_course_type=env_config.get("obstacle_course_type", "barn"),
        obstacle_shape=env_config.get("obstacle_shape", "cylinder"),
        obstacle_difficulty=env_config.get("obstacle_difficulty", 0.5),
        use_curriculum=env_config.get("use_curriculum", False),  # Disable for preview
        num_obstacles_min=env_config.get("num_obstacles_min", 8),
        num_obstacles_max=env_config.get("num_obstacles_max", 30),
        cylinder_radius_min=env_config.get("cylinder_radius_min", 0.15),
        cylinder_radius_max=env_config.get("cylinder_radius_max", 0.35),
        cylinder_height_min=env_config.get("cylinder_height_min", 0.5),
        cylinder_height_max=env_config.get("cylinder_height_max", 1.0),
        course_width=env_config.get("course_width", 7.0),
        course_length=env_config.get("course_length", 18.0),
        obstacle_spawn_x_min=env_config.get("obstacle_spawn_x_min", 2.0),
        obstacle_spawn_x_max=env_config.get("obstacle_spawn_x_max", 16.0),
        min_passage_width=env_config.get("min_passage_width", 0.9),
        max_passage_width=env_config.get("max_passage_width", 2.0),
        obstacle_min_spawn_distance=env_config.get("obstacle_min_spawn_distance", 1.5),
        obstacle_waypoint_clearance=env_config.get("obstacle_waypoint_clearance", 1.0),
        randomize_obstacles_on_reset=True,
        randomize_obstacle_colors=True,
    )

    env = DifferentialDriveEnv(cfg=env_cfg, headless=False)

    print("\n" + "=" * 60)
    print("Preview Mode - Press SPACE in Isaac Sim to reset")
    print("Close Isaac Sim window or Ctrl+C to exit")
    print("=" * 60)

    # Wait for GUI
    print("\nInitializing GUI...")
    for _ in range(60):
        if not env.sim_app.is_running():
            break
        env._world.step(render=True)
        time.sleep(0.05)

    reset_count = 0
    try:
        while env.sim_app.is_running():
            # Initial reset
            obs, info = env.reset()
            reset_count += 1
            print(f"\n--- Reset {reset_count}/{args.resets} ---")
            print(f"Waypoints: {len(env._waypoints)}")
            for i, wp in enumerate(env._waypoints):
                print(f"  WP{i+1}: ({wp[0]:.1f}, {wp[1]:.1f})")

            if reset_count >= args.resets:
                print("\nMax resets reached. Press Ctrl+C or close window.")

            # Just render without taking actions
            step = 0
            while env.sim_app.is_running():
                # Take zero action (robot stays still)
                action = np.array([0.0, 0.0], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1

                # Auto-reset after some time
                if step > 300:  # ~5 seconds at 60Hz
                    if reset_count < args.resets:
                        break
                    else:
                        step = 0  # Keep showing last config

                time.sleep(0.016)  # ~60 FPS

    except KeyboardInterrupt:
        print("\n\nExiting...")

    finally:
        env.close()
        print("Done!")

    return 0


if __name__ == "__main__":
    sys.exit(main())

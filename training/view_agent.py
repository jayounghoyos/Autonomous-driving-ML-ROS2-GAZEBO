#!/usr/bin/env python3
"""
View Trained Agent in Isaac Sim GUI.

This script loads a trained model and runs it with visualization.
It uses a simpler approach that works better with Isaac Sim's display.

Usage:
    $ISAAC_PYTHON training/view_agent.py --model models/ppo/xxx/final_model.zip
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View trained agent")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--episodes", type=int, default=5, help="Episodes to run")
    parser.add_argument("--slow", action="store_true", help="Slow motion (0.5x speed)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        # Try adding .zip extension
        model_path = Path(str(args.model) + ".zip")
        if not model_path.exists():
            print(f"Error: Model not found at {args.model}")
            return 1

    print("=" * 60)
    print("Leatherback Agent Viewer")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {args.episodes}")
    print("=" * 60)

    # Initialize Isaac Sim WITH display
    print("\nStarting Isaac Sim with GUI...")
    from isaacsim import SimulationApp

    # Enable GUI
    sim_app = SimulationApp({
        "headless": False,
        "width": 1280,
        "height": 720,
        "window_width": 1440,
        "window_height": 900,
    })

    # Now import the rest
    import numpy as np
    import torch
    from stable_baselines3 import PPO

    # Import Isaac modules after SimulationApp
    from isaacsim.core.api import World
    from isaacsim.storage.native import get_assets_root_path
    import isaacsim.core.utils.stage as stage_utils
    from pxr import UsdLux
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import VisualSphere

    print("Creating world...")
    world = World()
    world.scene.add_default_ground_plane()
    stage = world.stage

    # Add lighting
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(2000.0)

    # Load robot
    print("Loading Leatherback robot...")
    assets_root = get_assets_root_path()
    robot_path = f"{assets_root}/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
    stage_utils.add_reference_to_stage(robot_path, "/World/Leatherback")

    robot = Articulation(prim_path="/World/Leatherback", name="leatherback")
    world.scene.add(robot)

    # Goal marker
    goal_marker = VisualSphere(
        prim_path="/World/Goal",
        name="goal",
        radius=0.3,
        color=np.array([0.0, 1.0, 0.0]),
        position=np.array([5.0, 0.0, 0.3]),
    )
    world.scene.add(goal_marker)

    # Initialize
    world.reset()

    # Get joint indices
    throttle_names = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left",
    ]
    steering_names = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    throttle_indices = [robot.get_dof_index(name) for name in throttle_names]
    steering_indices = [robot.get_dof_index(name) for name in steering_names]

    # Load model
    print(f"Loading model from {model_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO.load(str(model_path), device=device)
    print(f"Model loaded (device: {device})")

    # Configuration
    max_throttle = 30.0
    max_steering = 0.5
    goal_tolerance = 0.5
    waypoint_spacing = 4.0

    def get_observation(robot, goal_pos, prev_action):
        """Get observation for the model."""
        pos, quat = robot.get_world_pose()
        w, x, y, z = quat
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        goal_vec = goal_pos[:2] - pos[:2]
        distance = np.linalg.norm(goal_vec)
        target_heading = np.arctan2(goal_vec[1], goal_vec[0])
        heading_error = np.arctan2(
            np.sin(target_heading - heading),
            np.cos(target_heading - heading)
        )

        return {
            "vector": np.array([
                distance,
                np.cos(heading_error),
                np.sin(heading_error),
                prev_action[0],
                prev_action[1],
            ], dtype=np.float32)
        }

    def apply_action(robot, action, throttle_idx, steering_idx):
        """Apply action to robot."""
        throttle = float(np.clip(action[0], -1.0, 1.0)) * max_throttle
        steering = float(np.clip(action[1], -1.0, 1.0)) * max_steering

        throttle = float(np.clip(throttle, -30.0, 30.0))
        steering = float(np.clip(steering, -0.5, 0.5))

        robot.set_joint_velocities(
            np.full(4, throttle, dtype=np.float32),
            joint_indices=throttle_idx
        )
        robot.set_joint_positions(
            np.full(2, steering, dtype=np.float32),
            joint_indices=steering_idx
        )

    print("\n" + "=" * 60)
    print("Starting visualization - Close window to exit")
    print("=" * 60)

    try:
        for episode in range(args.episodes):
            # Reset robot
            robot.set_world_pose(
                position=np.array([0.0, 0.0, 0.05]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
            robot.set_joint_velocities(
                np.zeros(len(throttle_indices) + len(steering_indices)),
                joint_indices=throttle_indices + steering_indices
            )

            # Generate waypoints
            waypoints = []
            for i in range(5):
                x = (i + 1) * waypoint_spacing
                y = np.random.uniform(-2.0, 2.0)
                waypoints.append(np.array([x, y, 0.3]))

            current_wp = 0
            goal_marker.set_world_pose(position=waypoints[current_wp])

            prev_action = np.zeros(2, dtype=np.float32)
            step = 0

            print(f"\nEpisode {episode + 1}/{args.episodes}")

            while sim_app.is_running() and step < 1000:
                # Get observation
                obs = get_observation(robot, waypoints[current_wp], prev_action)

                # Get action from model
                action, _ = model.predict(obs, deterministic=True)

                # Apply action
                apply_action(robot, action, throttle_indices, steering_indices)
                prev_action = action.copy()

                # Step simulation
                world.step(render=True)

                if args.slow:
                    time.sleep(0.03)

                # Check waypoint
                distance = obs["vector"][0]
                if distance < goal_tolerance:
                    current_wp += 1
                    print(f"  Waypoint {current_wp}/5 reached!")
                    if current_wp >= len(waypoints):
                        print("  All waypoints completed!")
                        break
                    goal_marker.set_world_pose(position=waypoints[current_wp])

                # Check bounds
                pos, _ = robot.get_world_pose()
                if np.linalg.norm(pos[:2]) > 25.0 or pos[2] < -0.5:
                    print("  Out of bounds!")
                    break

                step += 1

            if not sim_app.is_running():
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print("\nClosing simulation...")
    sim_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

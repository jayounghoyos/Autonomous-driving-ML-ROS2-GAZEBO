"""
Configuration for Leatherback Navigation Environment.

This module defines the configuration dataclass for the Leatherback
autonomous vehicle navigation task using Isaac Lab.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LeatherbackEnvCfg:
    """Configuration for the Leatherback navigation environment.

    This configuration follows Isaac Lab patterns for RL environments,
    allowing easy modification of training parameters.
    """

    # =========================================================================
    # Environment Settings
    # =========================================================================

    # Number of parallel environments (GPU accelerated)
    num_envs: int = 4096

    # Environment spacing in meters (for parallel envs)
    env_spacing: float = 20.0

    # Episode length in seconds
    episode_length_s: float = 60.0

    # Physics simulation timestep (matches Isaac Sim default)
    physics_dt: float = 1.0 / 60.0

    # Rendering decimation (render every N physics steps)
    decimation: int = 4

    # =========================================================================
    # Robot Configuration
    # =========================================================================

    # Robot USD asset path (uses NVIDIA cloud assets)
    # Set to None to use ISAAC_NUCLEUS_DIR/Robots/NVIDIA/Leatherback/leatherback.usd
    robot_usd_path: str | None = None

    # Initial robot position [x, y, z]
    robot_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.05)

    # Randomize initial heading
    randomize_heading: bool = True

    # =========================================================================
    # Navigation Task Configuration
    # =========================================================================

    # Goal tolerance in meters
    goal_tolerance: float = 0.5

    # Number of waypoints per episode
    num_waypoints: int = 10

    # Waypoint spacing in meters
    waypoint_spacing: float = 5.0

    # Lateral waypoint variation range
    waypoint_lateral_range: float = 3.0

    # Arena boundary radius (for geofence)
    arena_radius: float = 12.0

    # =========================================================================
    # Obstacle Configuration
    # =========================================================================

    # Number of obstacles to spawn
    num_obstacles: int = 5

    # Obstacle spawn radius (min/max from origin)
    obstacle_spawn_radius_min: float = 5.0
    obstacle_spawn_radius_max: float = 10.0

    # Obstacle size [width, depth, height]
    obstacle_size: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # =========================================================================
    # Observation Configuration
    # =========================================================================

    # Use RGB camera
    use_camera: bool = False

    # Camera resolution
    camera_resolution: tuple[int, int] = (64, 64)

    # Camera position relative to chassis [x, y, z]
    camera_position: tuple[float, float, float] = (0.5, 0.0, 1.0)

    # Use LiDAR
    use_lidar: bool = False

    # LiDAR position relative to chassis [x, y, z]
    lidar_position: tuple[float, float, float] = (0.0, 0.0, 1.2)

    # LiDAR number of points (resampled)
    lidar_num_points: int = 360

    # LiDAR max range in meters
    lidar_max_range: float = 20.0

    # Vector observation size (distance, cos_heading, sin_heading, prev_throttle, prev_steering)
    vector_obs_size: int = 5

    # =========================================================================
    # Action Configuration
    # =========================================================================

    # Max throttle velocity (rad/s for wheel joints) - reduced for stability
    max_throttle: float = 30.0

    # Max steering angle (radians) - kept within safe PhysX limits
    max_steering: float = 0.5

    # =========================================================================
    # Reward Configuration
    # =========================================================================

    # Progress reward scale (per meter of progress)
    reward_progress_scale: float = 10.0

    # Goal reached bonus
    reward_goal_bonus: float = 50.0

    # Time penalty per step
    reward_time_penalty: float = -0.05

    # Heading alignment reward scale
    reward_heading_scale: float = 0.5

    # Collision/boundary penalty
    reward_collision_penalty: float = -50.0

    # =========================================================================
    # Termination Conditions
    # =========================================================================

    # Terminate if robot flips (z-axis angle threshold)
    flip_threshold: float = 0.3  # cos(~70 degrees)

    # Terminate if robot falls below ground
    fall_threshold: float = 0.0

    # =========================================================================
    # Physics Configuration
    # =========================================================================

    # Solver type for PhysX
    solver_type: Literal["TGS", "PGS"] = "TGS"

    # Disable self-collisions for robot
    disable_self_collisions: bool = True

    # Wheel joint physics
    throttle_stiffness: float = 0.0
    throttle_damping: float = 10.0
    steering_stiffness: float = 10000.0
    steering_damping: float = 1000.0

    # =========================================================================
    # Joint Names (Leatherback-specific)
    # =========================================================================

    throttle_joint_names: tuple[str, ...] = (
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left",
    )

    steering_joint_names: tuple[str, ...] = (
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    )


# Pre-configured variants for common use cases
@dataclass
class LeatherbackEnvCfgHeadless(LeatherbackEnvCfg):
    """Configuration for headless training (no sensors, max performance)."""
    use_camera: bool = False
    use_lidar: bool = False
    num_envs: int = 4096


@dataclass
class LeatherbackEnvCfgWithSensors(LeatherbackEnvCfg):
    """Configuration with camera and LiDAR enabled."""
    use_camera: bool = True
    use_lidar: bool = True
    num_envs: int = 256  # Reduced for sensor overhead


@dataclass
class LeatherbackEnvCfgDebug(LeatherbackEnvCfg):
    """Configuration for debugging with single environment."""
    num_envs: int = 1
    use_camera: bool = True
    use_lidar: bool = True

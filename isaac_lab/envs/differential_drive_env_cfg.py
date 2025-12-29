"""
Configuration for 4-Wheel Differential Drive (Skid-Steer) Robot Environment.

This module defines the configuration for a 4-wheel differential drive robot
using the Clearpath Jackal model from Isaac Sim assets.

For sim2real with skid-steer robots, the wheelDistance parameter is scaled
to account for tire slippage during turns (typically 4-5x actual value).

Supported robots:
- Clearpath Jackal (default): /Isaac/Robots/Clearpath/Jackal/jackal.usd
- Clearpath Dingo: /Isaac/Robots/Clearpath/Dingo/dingo.usd
- AgileX Limo: /Isaac/Robots/AgilexRobotics/limo/limo.usd
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class DifferentialDriveEnvCfg:
    """Configuration for 4-wheel differential drive robot navigation.

    Default robot: Clearpath Jackal
    - 4 wheels, skid-steer drive
    - 2 built-in cameras, 1 IMU
    - Well-documented specs for sim2real

    Jackal Specifications:
    - wheel_radius: 0.098m
    - track_width: 0.37558m (wheel separation)
    - Robot size: ~508mm x 430mm x 250mm
    """

    # =========================================================================
    # Robot Selection
    # =========================================================================

    # Robot USD path (None = use default Jackal from Isaac assets)
    robot_usd_path: str | None = None

    # Robot type for auto-configuration
    robot_type: Literal["jackal", "dingo", "limo", "custom"] = "jackal"

    # =========================================================================
    # Clearpath Jackal Geometry (from official specs)
    # =========================================================================

    # Wheel radius in meters (Jackal: 0.098m)
    wheel_radius: float = 0.098

    # Track width - distance between left and right wheel centers (Jackal: 0.37558m)
    track_width: float = 0.37558

    # Wheelbase - distance between front and rear axles (Jackal: ~0.262m)
    wheelbase: float = 0.262

    # Effective wheel distance for skid-steer kinematics
    # For 4-wheel skid-steer, this should be ~4.2x actual track_width
    # to account for tire slippage during turns (from NVIDIA forums)
    skid_steer_correction: float = 4.2

    # Robot mass in kg (Jackal: ~17kg)
    robot_mass: float = 17.0

    # Chassis prim path within the robot USD
    chassis_prim_name: str = "base_link"

    # =========================================================================
    # Environment Settings
    # =========================================================================

    num_envs: int = 1
    env_spacing: float = 20.0
    episode_length_s: float = 120.0
    physics_dt: float = 1.0 / 60.0
    decimation: int = 4

    # =========================================================================
    # Navigation Task
    # =========================================================================

    goal_tolerance: float = 0.5
    num_waypoints: int = 5
    waypoint_spacing: float = 4.0
    waypoint_lateral_range: float = 3.0
    arena_radius: float = 25.0

    # =========================================================================
    # Progressive Goal Curriculum (v3)
    # =========================================================================
    # Progressively increase goal distance and number of waypoints
    # Stage 1: 1 close goal (learn basic navigation)
    # Stage 2: 2 goals (learn sequencing)
    # Stage 3: 3 goals (full course)
    # Stage 4: 3 goals with max lateral randomness

    use_progressive_goals: bool = False

    # Stage 1: Episodes 0-200
    stage1_episodes: int = 200
    stage1_num_waypoints: int = 1
    stage1_goal_distance_min: float = 4.0
    stage1_goal_distance_max: float = 6.0
    stage1_lateral_range: float = 1.0

    # Stage 2: Episodes 200-400
    stage2_episodes: int = 400
    stage2_num_waypoints: int = 2
    stage2_goal_spacing: float = 5.0
    stage2_lateral_range: float = 1.5

    # Stage 3: Episodes 400-600
    stage3_episodes: int = 600
    stage3_num_waypoints: int = 3
    stage3_goal_spacing: float = 5.0
    stage3_lateral_range: float = 2.0

    # Stage 4: Episodes 600+
    stage4_num_waypoints: int = 3
    stage4_goal_spacing: float = 5.0
    stage4_lateral_range: float = 2.5

    # =========================================================================
    # BARN-Style Obstacle Course Configuration
    # =========================================================================
    # Inspired by BARN (Benchmark for Autonomous Robot Navigation)
    # https://www.cs.utexas.edu/~xiao/BARN/BARN.html
    #
    # Features:
    # - Cylindrical obstacles (more realistic)
    # - Difficulty-based generation (0.0 easy → 1.0 hard)
    # - Narrow passages and corridors
    # - Curriculum learning support
    # =========================================================================

    # Obstacle course type: "random", "corridor", "maze", "barn"
    obstacle_course_type: Literal["random", "corridor", "maze", "barn"] = "barn"

    # Difficulty level (0.0 = easy/sparse, 1.0 = hard/dense)
    obstacle_difficulty: float = 0.5

    # Use curriculum learning (start easy, get harder)
    use_curriculum: bool = True
    curriculum_start_difficulty: float = 0.2
    curriculum_end_difficulty: float = 0.9
    curriculum_episodes_to_max: int = 500

    # Obstacle counts (interpolated by difficulty)
    num_obstacles_min: int = 5       # At difficulty 0.0
    num_obstacles_max: int = 40      # At difficulty 1.0

    # Obstacle shape: "cylinder", "cube", "mixed"
    obstacle_shape: Literal["cylinder", "cube", "mixed"] = "cylinder"

    # Cylinder obstacle dimensions
    cylinder_radius_min: float = 0.15   # Thin pillars
    cylinder_radius_max: float = 0.4    # Thick pillars
    cylinder_height_min: float = 0.5    # Short
    cylinder_height_max: float = 1.2    # Tall

    # Cube obstacle dimensions (if using cubes)
    obstacle_size_min: tuple[float, float, float] = (0.3, 0.3, 0.5)
    obstacle_size_max: tuple[float, float, float] = (0.8, 0.8, 1.0)

    # Course dimensions (5m x 5m like BARN)
    course_width: float = 8.0       # Width of obstacle course (Y axis)
    course_length: float = 20.0     # Length of obstacle course (X axis)

    # Spawn area (obstacles spawn in this region)
    obstacle_spawn_x_min: float = 2.0    # Start after robot spawn
    obstacle_spawn_x_max: float = 18.0   # End before final goal

    # Passage width (for corridor/maze modes)
    min_passage_width: float = 0.8   # Robot width ~0.5m, need clearance
    max_passage_width: float = 2.0   # Wider passages at low difficulty

    # Obstacle colors
    randomize_obstacle_colors: bool = True
    randomize_obstacles_on_reset: bool = True

    # Safety: minimum distance from robot spawn and waypoints
    obstacle_min_spawn_distance: float = 1.5      # From robot start
    obstacle_waypoint_clearance: float = 1.0      # From waypoints

    # =========================================================================
    # Sensors
    # =========================================================================

    use_camera: bool = False
    camera_resolution: tuple[int, int] = (64, 64)
    camera_position: tuple[float, float, float] = (0.15, 0.0, 0.15)  # Front-mounted

    use_lidar: bool = False
    lidar_position: tuple[float, float, float] = (0.0, 0.0, 0.20)  # Top-mounted
    lidar_num_points: int = 180
    lidar_max_range: float = 10.0  # Smaller robot = shorter range

    vector_obs_size: int = 5

    # Multi-modal goal observation size (for camera+lidar mode)
    # [dist_norm, sin(heading), cos(heading), progress, lin_vel, ang_vel]
    goal_obs_size: int = 6

    # =========================================================================
    # Motor Configuration (Jackal specs)
    # =========================================================================

    # Max wheel angular velocity (rad/s)
    # Jackal max speed: 2.0 m/s -> omega = v/r = 2.0/0.098 = 20.4 rad/s
    max_wheel_velocity: float = 20.0

    # Max linear velocity (Jackal: 2.0 m/s)
    max_linear_velocity: float = 2.0  # m/s

    # Max angular velocity for turns (Jackal: ~4.0 rad/s)
    max_angular_velocity: float = 4.0  # rad/s

    # Motor drive parameters
    wheel_stiffness: float = 0.0  # Velocity control
    wheel_damping: float = 100.0  # Higher damping for stability

    # =========================================================================
    # Jackal Joint Names (from USD)
    # =========================================================================

    # Wheel joint names in the Jackal USD
    wheel_joint_names: tuple[str, ...] = (
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "rear_left_wheel_joint",
        "rear_right_wheel_joint",
    )

    # =========================================================================
    # CONDITIONAL DENSE REWARD CONFIGURATION
    # =========================================================================
    #
    # PRINCIPLE: Reward behaviors ONLY when making progress
    # - Progress toward goal (always active - core driver)
    # - Heading bonus (ONLY if making progress - prevents spinning exploit)
    # - Velocity bonus (ONLY if heading is good - prevents wrong-direction exploit)
    # - Smooth actions (ONLY if making progress - prevents smooth-but-stuck exploit)
    # - Obstacle avoidance (always active - safety)
    # - Stuck penalty (always active - anti-exploitation)
    #
    # This encourages good autonomous driving behavior while preventing hacking.
    # =========================================================================

    # CORE: Progress reward (THE PRIMARY DRIVER - always active)
    reward_progress_scale: float = 30.0       # Strong: actual movement toward goal
    reward_away_penalty_scale: float = 15.0   # Weaker: allow some exploration

    # CONDITIONAL: Heading bonus (only when making progress)
    reward_heading_scale: float = 0.5         # Bonus for facing goal
    progress_gate: float = 0.01               # Min progress (m) to get heading bonus

    # CONDITIONAL: Velocity bonus (only when heading is good)
    reward_velocity_scale: float = 0.3        # Bonus for moving at speed
    heading_gate: float = 0.7                 # Min cos(heading_error) for velocity bonus (~45°)

    # CONDITIONAL: Smooth action (only when making progress)
    reward_smooth_scale: float = 0.1          # Small bonus for smooth control

    # FORWARD PREFERENCE: Penalize reverse motion
    reward_reverse_penalty: float = -0.3      # Penalty for going backwards

    # GOAL: Sparse rewards for task completion
    reward_waypoint_bonus: float = 200.0      # Big bonus for reaching waypoint
    reward_all_waypoints_bonus: float = 500.0 # Extra for completing all
    reward_collision_penalty: float = -150.0  # Strong penalty for failure

    # SAFETY: LiDAR-based obstacle avoidance (always active)
    reward_obstacle_danger_zone: float = 2.0  # Distance to start penalizing
    reward_obstacle_penalty_max: float = 8.0  # Max penalty when very close

    # ANTI-STUCK: Prevent exploitation (always active)
    reward_stuck_penalty: float = -2.0        # Penalty when not moving
    stuck_threshold_steps: int = 30           # Steps before considered stuck
    stuck_movement_threshold: float = 0.05    # Min movement required (meters)

    # =========================================================================
    # Termination Conditions
    # =========================================================================

    flip_threshold: float = 0.3
    fall_threshold: float = -0.1

    # =========================================================================
    # Physics
    # =========================================================================

    solver_type: Literal["TGS", "PGS"] = "TGS"
    disable_self_collisions: bool = True

    # Ground friction (important for skid-steer)
    ground_friction: float = 0.8

    # Wheel friction (affects turning behavior)
    wheel_friction: float = 1.0


@dataclass
class DifferentialDriveEnvCfgFullSensors(DifferentialDriveEnvCfg):
    """Configuration with camera and LiDAR enabled."""
    use_camera: bool = True
    use_lidar: bool = True


@dataclass
class DifferentialDriveEnvCfgTest(DifferentialDriveEnvCfg):
    """Configuration for testing with minimal obstacles."""
    num_obstacles_min: int = 0
    num_obstacles_max: int = 2
    randomize_obstacles_on_reset: bool = False


@dataclass
class DifferentialDriveEnvCfgBARN(DifferentialDriveEnvCfg):
    """BARN-style obstacle course configuration.

    Based on: https://www.cs.utexas.edu/~xiao/BARN/BARN.html
    - 300 environments with varying difficulty
    - Cylindrical obstacles
    - Narrow passages
    - Curriculum learning
    """
    # Enable LiDAR for obstacle detection
    use_lidar: bool = True
    lidar_num_points: int = 180
    lidar_max_range: float = 10.0

    # BARN-style course
    obstacle_course_type: Literal["random", "corridor", "maze", "barn"] = "barn"
    obstacle_shape: Literal["cylinder", "cube", "mixed"] = "cylinder"

    # Difficulty and curriculum
    obstacle_difficulty: float = 0.3
    use_curriculum: bool = True
    curriculum_start_difficulty: float = 0.2
    curriculum_end_difficulty: float = 0.8
    curriculum_episodes_to_max: int = 300

    # More obstacles for challenge
    num_obstacles_min: int = 8
    num_obstacles_max: int = 35

    # Tighter course
    course_width: float = 6.0
    course_length: float = 18.0

    # Navigation
    num_waypoints: int = 3
    waypoint_spacing: float = 5.0
    waypoint_lateral_range: float = 2.0
    goal_tolerance: float = 0.8

    # Shorter episodes (need to be efficient)
    episode_length_s: float = 60.0


@dataclass
class DifferentialDriveEnvCfgMultiModal(DifferentialDriveEnvCfg):
    """Multi-modal configuration with Camera + LiDAR + Goal info.

    This configuration enables sensor fusion for robust navigation:
    - RGB Camera (84x84): Visual features, semantic understanding
    - LiDAR (180 points): Precise distance measurements
    - Goal Info (6 values): Navigation direction and state

    Observation space is a Dict compatible with SB3 MultiInputPolicy:
    {
        "camera": Box(0, 255, (84, 84, 3), uint8),
        "lidar": Box(0, 10, (180,), float32),
        "goal": Box(-1, 1, (6,), float32)
    }
    """

    # =========================================================================
    # Sensor Configuration
    # =========================================================================

    # Enable both sensors
    use_camera: bool = True
    use_lidar: bool = True

    # Camera: 84x84 RGB (standard for vision RL)
    camera_resolution: tuple[int, int] = (84, 84)
    camera_position: tuple[float, float, float] = (0.2, 0.0, 0.2)  # Front-mounted, slightly elevated
    camera_fov: float = 90.0  # Field of view in degrees

    # LiDAR: 180 points, 10m range
    lidar_num_points: int = 180
    lidar_max_range: float = 10.0
    lidar_position: tuple[float, float, float] = (0.0, 0.0, 0.25)

    # Goal observation size: [dist, sin, cos, progress, lin_vel, ang_vel]
    goal_obs_size: int = 6

    # =========================================================================
    # BARN-Style Obstacle Course
    # =========================================================================

    obstacle_course_type: Literal["random", "corridor", "maze", "barn"] = "barn"
    obstacle_shape: Literal["cylinder", "cube", "mixed"] = "mixed"

    # Difficulty and curriculum
    use_curriculum: bool = True
    curriculum_start_difficulty: float = 0.15
    curriculum_end_difficulty: float = 0.75
    curriculum_episodes_to_max: int = 600

    # Obstacle counts
    num_obstacles_min: int = 8
    num_obstacles_max: int = 30

    # Course dimensions
    course_width: float = 8.0
    course_length: float = 20.0

    # =========================================================================
    # Progressive Goals (same as v3)
    # =========================================================================

    use_progressive_goals: bool = True

    # Stage 1: 1 close goal
    stage1_episodes: int = 200
    stage1_num_waypoints: int = 1
    stage1_goal_distance_min: float = 4.0
    stage1_goal_distance_max: float = 6.0
    stage1_lateral_range: float = 1.0

    # Stage 2: 2 goals
    stage2_episodes: int = 400
    stage2_num_waypoints: int = 2
    stage2_goal_spacing: float = 5.0
    stage2_lateral_range: float = 1.5

    # Stage 3: 3 goals
    stage3_episodes: int = 600
    stage3_num_waypoints: int = 3
    stage3_goal_spacing: float = 5.0
    stage3_lateral_range: float = 2.0

    # Stage 4: Full randomness
    stage4_num_waypoints: int = 3
    stage4_goal_spacing: float = 5.0
    stage4_lateral_range: float = 2.5

    # =========================================================================
    # Navigation
    # =========================================================================

    num_waypoints: int = 3
    waypoint_spacing: float = 5.0
    waypoint_lateral_range: float = 2.0
    goal_tolerance: float = 0.8

    # Episode length
    episode_length_s: float = 90.0  # Longer for visual learning

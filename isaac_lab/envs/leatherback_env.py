"""
Leatherback RL Environment for Isaac Sim 5.1.0 + Isaac Lab.

This module implements a Gymnasium-compatible environment for training
autonomous navigation with the NVIDIA Leatherback vehicle.

Based on Isaac Lab patterns for compatibility with Stable-Baselines3.
"""

from __future__ import annotations

import os
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# Handle both module import and direct execution
try:
    from .leatherback_env_cfg import LeatherbackEnvCfg
except ImportError:
    from leatherback_env_cfg import LeatherbackEnvCfg


class LeatherbackEnv(gym.Env):
    """Leatherback autonomous vehicle navigation environment.

    This environment uses Isaac Sim 5.1.0 for physics simulation and
    follows Gymnasium API for compatibility with RL libraries.

    Args:
        cfg: Environment configuration dataclass.
        headless: Whether to run without GUI.
        render_mode: Gymnasium render mode (unused, kept for API compatibility).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: LeatherbackEnvCfg | None = None,
        headless: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()

        # Use default config if none provided
        self.cfg = cfg if cfg is not None else LeatherbackEnvCfg()
        self.render_mode = render_mode
        self._headless = headless

        print(f"Initializing Leatherback environment (headless={headless})...")

        # Initialize Isaac Sim - must be done before any other imports
        from isaacsim import SimulationApp

        self._sim_app = SimulationApp({"headless": headless})

        # Import Isaac modules after SimulationApp initialization
        self._import_isaac_modules()

        # Create world and scene
        self._setup_world()

        # Load robot
        self._load_robot()

        # Configure physics
        self._configure_physics()

        # Setup sensors (optional)
        self._setup_sensors()

        # Spawn obstacles
        self._spawn_obstacles()

        # Create goal marker
        self._create_goal_marker()

        # Initialize world
        self._world.reset()

        # Get joint indices after world initialization
        self._get_joint_indices()

        # Define spaces
        self._define_spaces()

        # Initialize state variables
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._prev_distance = 0.0
        self._waypoints: np.ndarray | None = None
        self._current_waypoint_idx = 0

        print("Leatherback environment initialized successfully!")

    def _import_isaac_modules(self) -> None:
        """Import Isaac Sim modules after SimulationApp is created."""
        # Core APIs
        from isaacsim.core.api import World
        from isaacsim.storage.native import get_assets_root_path
        import isaacsim.core.utils.stage as stage_utils
        import isaacsim.core.utils.prims as prim_utils
        import isaacsim.core.utils.xforms as xforms_utils

        # USD/PhysX
        from pxr import UsdLux, UsdPhysics, PhysxSchema

        # Core objects (use omni.isaac.core for Articulation)
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.objects import VisualSphere, FixedCuboid

        # Ackermann controller for proper vehicle control
        from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController
        from isaacsim.core.utils.types import ArticulationAction

        # Store for later use
        self._World = World
        self._get_assets_root_path = get_assets_root_path
        self._stage_utils = stage_utils
        self._prim_utils = prim_utils
        self._xforms_utils = xforms_utils
        self._UsdLux = UsdLux
        self._UsdPhysics = UsdPhysics
        self._PhysxSchema = PhysxSchema
        self._Articulation = Articulation
        self._VisualSphere = VisualSphere
        self._FixedCuboid = FixedCuboid
        self._AckermannController = AckermannController
        self._ArticulationAction = ArticulationAction

    def _setup_world(self) -> None:
        """Create the simulation world."""
        self._world = self._World()
        self._world.scene.add_default_ground_plane()
        self._stage = self._world.stage

        # Add lighting
        dome_light = self._UsdLux.DomeLight.Define(self._stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(2000.0)
        print("  World and lighting created")

    def _load_robot(self) -> None:
        """Load the Leatherback robot from assets."""
        # Determine asset path
        if self.cfg.robot_usd_path is not None:
            leatherback_path = self.cfg.robot_usd_path
        else:
            assets_root = self._get_assets_root_path()
            if assets_root is None:
                raise RuntimeError(
                    "Isaac Sim assets not configured! "
                    "Run Isaac Sim once to download assets or set ISAAC_NUCLEUS_DIR."
                )
            leatherback_path = f"{assets_root}/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"

        # Add robot to stage
        self._stage_utils.add_reference_to_stage(leatherback_path, "/World/Leatherback")
        print(f"  Robot loaded from: {leatherback_path}")

        # Create Articulation wrapper
        self._robot = self._Articulation(
            prim_path="/World/Leatherback",
            name="leatherback",
        )
        self._world.scene.add(self._robot)

    def _configure_physics(self) -> None:
        """Configure physics parameters for stable simulation."""
        # Disable self-collisions
        if self.cfg.disable_self_collisions:
            prim = self._stage.GetPrimAtPath("/World/Leatherback")
            if prim.IsValid():
                physx_articulation = self._PhysxSchema.PhysxArticulationAPI.Apply(prim)
                physx_articulation.GetEnabledSelfCollisionsAttr().Set(False)
                print("  Self-collisions disabled")

        # Configure solver
        try:
            self._world.get_physics_context().set_solver_type(self.cfg.solver_type)
        except AttributeError:
            # Fallback for different API versions
            scene_prim = self._stage.GetPrimAtPath("/World/PhysicsScene")
            if scene_prim.IsValid():
                physx_scene = self._PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
                physx_scene.GetSolverTypeAttr().Set(self.cfg.solver_type)
        print(f"  Physics solver: {self.cfg.solver_type}")

        # Configure joint drives
        self._configure_joint_drives()

    def _configure_joint_drives(self) -> None:
        """Configure drive parameters for throttle and steering joints."""
        joints_path = "/World/Leatherback/Joints"

        # Throttle joints (velocity control)
        for joint_name in self.cfg.throttle_joint_names:
            prim_path = f"{joints_path}/{joint_name}"
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                drive = self._UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(self.cfg.throttle_stiffness)
                drive.GetDampingAttr().Set(self.cfg.throttle_damping)

        # Steering joints (position control)
        for joint_name in self.cfg.steering_joint_names:
            prim_path = f"{joints_path}/{joint_name}"
            prim = self._stage.GetPrimAtPath(prim_path)
            if prim.IsValid():
                drive = self._UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(self.cfg.steering_stiffness)
                drive.GetDampingAttr().Set(self.cfg.steering_damping)

        print("  Joint drives configured")

    def _setup_sensors(self) -> None:
        """Setup camera and LiDAR sensors if enabled.

        Note: Leatherback in Isaac Sim 5.1.0 has 4 built-in cameras.
        We use the front camera for observations.
        """
        self._camera = None
        self._lidar = None

        if self.cfg.use_camera:
            try:
                from omni.isaac.sensor import Camera

                # Leatherback has built-in cameras - try to find them
                # Common camera paths in Leatherback USD:
                camera_paths = [
                    "/World/Leatherback/Sensors/Camera_Front",
                    "/World/Leatherback/Camera_Front",
                    "/World/Leatherback/Rigid_Bodies/Chassis/Camera_Front",
                    "/World/Leatherback/front_camera",
                ]

                camera_prim_path = None
                for path in camera_paths:
                    prim = self._stage.GetPrimAtPath(path)
                    if prim.IsValid():
                        camera_prim_path = path
                        print(f"  Found built-in camera at: {path}")
                        break

                if camera_prim_path:
                    # Use the built-in camera
                    self._camera = Camera(
                        prim_path=camera_prim_path,
                        name="front_camera",
                        frequency=30,
                        resolution=self.cfg.camera_resolution,
                    )
                else:
                    # Create new camera if no built-in found
                    camera_prim_path = "/World/Leatherback/Rigid_Bodies/Chassis/FrontCamera"
                    print(f"  No built-in camera found, creating at: {camera_prim_path}")

                    self._camera = Camera(
                        prim_path=camera_prim_path,
                        name="front_camera",
                        position=np.array(self.cfg.camera_position),
                        frequency=30,
                        resolution=self.cfg.camera_resolution,
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    )

                self._world.scene.add(self._camera)
                print(f"  Camera initialized: {self.cfg.camera_resolution}")
            except Exception as e:
                print(f"  Warning: Could not initialize camera: {e}")
                self._camera = None

        if self.cfg.use_lidar:
            try:
                from omni.isaac.sensor import LidarRtx

                self._lidar = LidarRtx(
                    prim_path="/World/Leatherback/Rigid_Bodies/Chassis/TopLidar",
                    name="lidar",
                    position=np.array(self.cfg.lidar_position),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                )
                self._world.scene.add(self._lidar)
                self._lidar.add_range_data_to_frame()
                print(f"  LiDAR initialized at {self.cfg.lidar_position}")
            except Exception as e:
                print(f"  Warning: Could not initialize LiDAR: {e}")
                self._lidar = None

    def _spawn_obstacles(self) -> None:
        """Spawn randomized obstacles in the environment."""
        self._obstacles = []
        self._obstacle_count = 0

        # Random number of obstacles
        num_obstacles = np.random.randint(
            self.cfg.num_obstacles_min,
            self.cfg.num_obstacles_max + 1
        )

        for i in range(num_obstacles):
            self._spawn_single_obstacle(i)

        self._obstacle_count = num_obstacles
        print(f"  {num_obstacles} randomized obstacles spawned")

    def _spawn_single_obstacle(self, index: int) -> None:
        """Spawn a single randomized obstacle."""
        # Random position avoiding robot spawn area
        max_attempts = 50
        for _ in range(max_attempts):
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(
                self.cfg.obstacle_spawn_radius_min,
                self.cfg.obstacle_spawn_radius_max,
            )
            pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])

            # Check minimum distance from robot spawn
            if np.linalg.norm(pos) >= self.cfg.obstacle_min_spawn_distance:
                break

        # Random size
        size = np.array([
            np.random.uniform(self.cfg.obstacle_size_min[0], self.cfg.obstacle_size_max[0]),
            np.random.uniform(self.cfg.obstacle_size_min[1], self.cfg.obstacle_size_max[1]),
            np.random.uniform(self.cfg.obstacle_size_min[2], self.cfg.obstacle_size_max[2]),
        ])

        # Random color
        if self.cfg.randomize_obstacle_colors:
            # Generate varied colors (avoiding pure green which is goal marker)
            color = np.array([
                np.random.uniform(0.3, 0.9),
                np.random.uniform(0.1, 0.5),
                np.random.uniform(0.1, 0.7),
            ])
        else:
            color = np.array([0.8, 0.2, 0.2])  # Default red

        obstacle = self._FixedCuboid(
            prim_path=f"/World/Obstacle_{index}",
            name=f"obstacle_{index}",
            position=np.array([pos[0], pos[1], size[2] / 2]),
            scale=size,
            color=color,
        )
        self._world.scene.add(obstacle)
        self._obstacles.append(obstacle)

    def _randomize_obstacles(self) -> None:
        """Re-randomize obstacle positions on episode reset.

        Note: Only changes positions, not scale, to avoid PhysX instability.
        """
        if not self.cfg.randomize_obstacles_on_reset:
            return

        for obstacle in self._obstacles:
            # Random position - keep obstacles away from spawn area
            max_attempts = 100
            for _ in range(max_attempts):
                angle = np.random.uniform(0, 2 * np.pi)
                # Minimum 5m from origin to avoid spawn collisions
                radius = np.random.uniform(
                    max(5.0, self.cfg.obstacle_spawn_radius_min),
                    self.cfg.obstacle_spawn_radius_max,
                )
                pos = np.array([radius * np.cos(angle), radius * np.sin(angle)])
                if np.linalg.norm(pos) >= 5.0:
                    break

            # Update position only (scale changes cause physics issues)
            try:
                obstacle.set_world_pose(
                    position=np.array([pos[0], pos[1], 0.75])
                )
            except Exception:
                pass

    def _create_goal_marker(self) -> None:
        """Create visual marker for current goal waypoint."""
        self._goal_marker = self._VisualSphere(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            radius=0.3,
            color=np.array([0.0, 1.0, 0.0]),
            position=np.array([100.0, 100.0, 0.3]),  # Start hidden
        )
        self._world.scene.add(self._goal_marker)

    def _get_joint_indices(self) -> None:
        """Get DOF indices for throttle and steering joints."""
        self._throttle_indices = [
            self._robot.get_dof_index(name) for name in self.cfg.throttle_joint_names
        ]
        self._steering_indices = [
            self._robot.get_dof_index(name) for name in self.cfg.steering_joint_names
        ]
        print(f"  Throttle indices: {self._throttle_indices}")
        print(f"  Steering indices: {self._steering_indices}")

        # Initialize Ackermann controller with Leatherback geometry
        # Leatherback specs: wheel_base=1.65m, track_width=1.25m, wheel_radius=0.25m
        self._ackermann_controller = self._AckermannController(
            name="leatherback_controller",
            wheel_base=1.65,
            track_width=1.25,
            front_wheel_radius=0.25,
            back_wheel_radius=0.25,
        )
        print("  Ackermann controller initialized")

    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        obs_dict = {
            "vector": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.cfg.vector_obs_size,),
                dtype=np.float32,
            ),
        }

        if self.cfg.use_camera:
            obs_dict["image"] = spaces.Box(
                low=0,
                high=255,
                shape=(*self.cfg.camera_resolution, 3),
                dtype=np.uint8,
            )

        if self.cfg.use_lidar:
            obs_dict["lidar"] = spaces.Box(
                low=0.0,
                high=self.cfg.lidar_max_range,
                shape=(self.cfg.lidar_num_points,),
                dtype=np.float32,
            )

        self.observation_space = spaces.Dict(obs_dict)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _generate_waypoints(self) -> np.ndarray:
        """Generate navigation waypoints for the episode."""
        waypoints = []
        for i in range(self.cfg.num_waypoints):
            x = (i + 1) * self.cfg.waypoint_spacing
            y = np.random.uniform(
                -self.cfg.waypoint_lateral_range,
                self.cfg.waypoint_lateral_range,
            )
            waypoints.append([x, y])
        return np.array(waypoints, dtype=np.float32)

    def _get_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get robot position and orientation."""
        return self._xforms_utils.get_world_pose(
            prim_path="/World/Leatherback/Rigid_Bodies/Chassis"
        )

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Compute current observation."""
        position, orientation = self._get_robot_pose()

        # Calculate heading from quaternion [w, x, y, z]
        w, x, y, z = orientation
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Get current waypoint
        current_waypoint = self._waypoints[self._current_waypoint_idx]

        # Calculate goal-relative observations
        goal_vector = current_waypoint - position[:2]
        distance = np.linalg.norm(goal_vector)
        target_heading = np.arctan2(goal_vector[1], goal_vector[0])
        heading_error = np.arctan2(
            np.sin(target_heading - heading), np.cos(target_heading - heading)
        )

        vector_obs = np.array(
            [
                distance,
                np.cos(heading_error),
                np.sin(heading_error),
                self._prev_action[0],
                self._prev_action[1],
            ],
            dtype=np.float32,
        )

        obs = {"vector": vector_obs}

        # Camera observation
        if self.cfg.use_camera and self._camera is not None:
            image = np.zeros((*self.cfg.camera_resolution, 3), dtype=np.uint8)
            try:
                rgba = self._camera.get_rgba()
                if rgba is not None and rgba.size > 0:
                    if len(rgba.shape) == 1:
                        expected_size = self.cfg.camera_resolution[0] * self.cfg.camera_resolution[1] * 4
                        if rgba.size == expected_size:
                            rgba = rgba.reshape((*self.cfg.camera_resolution, 4))
                    if len(rgba.shape) == 3 and rgba.shape[2] >= 3:
                        image = rgba[:, :, :3].astype(np.uint8)
            except Exception:
                pass
            obs["image"] = image

        # LiDAR observation
        if self.cfg.use_lidar and self._lidar is not None:
            lidar_data = np.zeros(self.cfg.lidar_num_points, dtype=np.float32)
            try:
                frame = self._lidar.get_current_frame()
                raw_data = frame.get("range") or frame.get("depth")
                if raw_data is not None and raw_data.size > 0:
                    if raw_data.size != self.cfg.lidar_num_points:
                        indices = np.linspace(
                            0, raw_data.size - 1, self.cfg.lidar_num_points, dtype=int
                        )
                        lidar_data = raw_data[indices]
                    else:
                        lidar_data = raw_data
                    lidar_data = np.clip(lidar_data, 0.0, self.cfg.lidar_max_range)
            except Exception:
                pass
            obs["lidar"] = lidar_data

        return obs

    def _apply_action(self, action: np.ndarray) -> None:
        """Apply throttle and steering action using Ackermann controller."""
        # Clamp actions to [-1, 1] range
        throttle_norm = float(np.clip(action[0], -1.0, 1.0))
        steering_norm = float(np.clip(action[1], -1.0, 1.0))

        # Convert to physical units
        # Forward velocity in rad/s (wheel angular velocity)
        forward_vel = throttle_norm * self.cfg.max_throttle
        # Steering angle in radians
        steering_angle = steering_norm * self.cfg.max_steering

        # Use Ackermann controller to compute proper wheel commands
        # forward() expects: [steering_angle, steering_velocity, forward_vel, acceleration, dt]
        controller_input = [
            steering_angle,       # Desired steering angle (rad)
            0.0,                  # Steering velocity (rad/s) - 0 for position control
            forward_vel,          # Forward velocity (rad/s)
            0.0,                  # Acceleration (not used)
            self.cfg.physics_dt,  # Delta time
        ]

        # Get articulation action from controller
        # Returns ArticulationAction with steering positions and wheel velocities
        ackermann_action = self._ackermann_controller.forward(controller_input)

        # Extract values from controller output
        # Ackermann controller outputs: [left_steer, right_steer] positions and [4 wheel] velocities
        steer_positions = ackermann_action.joint_positions
        wheel_velocities = ackermann_action.joint_velocities

        # Create full joint arrays for all DOFs
        num_dofs = self._robot.num_dof
        full_positions = np.zeros(num_dofs, dtype=np.float32)
        full_velocities = np.zeros(num_dofs, dtype=np.float32)

        # Map steering positions to correct indices
        if steer_positions is not None and len(steer_positions) >= 2:
            for i, idx in enumerate(self._steering_indices[:2]):
                full_positions[idx] = float(steer_positions[i])

        # Map wheel velocities to correct indices
        if wheel_velocities is not None and len(wheel_velocities) >= 4:
            for i, idx in enumerate(self._throttle_indices[:4]):
                full_velocities[idx] = float(wheel_velocities[i])

        # Create and apply the action with proper joint indices
        robot_action = self._ArticulationAction(
            joint_positions=full_positions,
            joint_velocities=full_velocities,
        )
        self._robot.apply_action(robot_action)

    def _calculate_reward(self, distance: float, terminated: bool) -> float:
        """Calculate reward for current step."""
        reward = 0.0

        # Progress reward
        progress = self._prev_distance - distance
        reward += progress * self.cfg.reward_progress_scale

        # Heading alignment (extracted from observation)
        # This is simplified - in practice you'd use the actual heading error

        # Time penalty
        reward += self.cfg.reward_time_penalty

        # Termination penalty
        if terminated:
            reward += self.cfg.reward_collision_penalty

        return reward

    def _check_termination(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> tuple[bool, bool, float]:
        """Check termination conditions.

        Returns:
            terminated: Episode ended due to success/failure
            truncated: Episode ended due to time limit
            bonus: Additional reward (goal bonus)
        """
        terminated = False
        truncated = False
        bonus = 0.0

        # Out of bounds
        dist_from_origin = np.linalg.norm(position[:2])
        if dist_from_origin > self.cfg.arena_radius:
            terminated = True
            print("  Out of bounds!")

        # Fall detection
        if position[2] < self.cfg.fall_threshold:
            terminated = True
            print("  Robot fell!")

        # Flip detection
        w, x, y, z = orientation
        z_axis_z = 1.0 - 2.0 * (x * x + y * y)
        if z_axis_z < self.cfg.flip_threshold:
            terminated = True
            print("  Robot flipped!")

        # Time limit (max steps)
        max_steps = int(self.cfg.episode_length_s / self.cfg.physics_dt / self.cfg.decimation)
        if self._step_count >= max_steps:
            truncated = True

        return terminated, truncated, bonus

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment to initial state.

        Uses world.reset() to properly reset physics state and avoid
        PhysX invalid transform errors.
        """
        super().reset(seed=seed)

        # Reset Ackermann controller
        self._ackermann_controller.reset()

        # Use world.reset() for proper physics state reset FIRST
        # This resets all articulations to their default states
        self._world.reset()

        # After world reset, randomize obstacles
        self._randomize_obstacles()

        # Generate new waypoints
        self._waypoints = self._generate_waypoints()
        self._current_waypoint_idx = 0

        # Update goal marker position AFTER world reset
        first_wp = self._waypoints[0]
        self._goal_marker.set_world_pose(
            position=np.array([first_wp[0], first_wp[1], 0.3])
        )

        # Apply zero velocities to ensure robot is stationary
        num_dofs = self._robot.num_dof
        zero_action = self._ArticulationAction(
            joint_positions=np.zeros(num_dofs, dtype=np.float32),
            joint_velocities=np.zeros(num_dofs, dtype=np.float32),
        )
        self._robot.apply_action(zero_action)

        # Step simulation to apply changes
        self._world.step(render=not self._headless)

        # Reset state variables
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)

        # Get initial observation
        obs = self._get_observation()
        self._prev_distance = obs["vector"][0]

        print(f"Reset: First waypoint at ({first_wp[0]:.1f}, {first_wp[1]:.1f})")

        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step."""
        self._step_count += 1
        self._prev_action = action.copy()

        # Apply action
        self._apply_action(action)

        # Step physics
        self._world.step(render=not self._headless)

        # Get state
        position, orientation = self._get_robot_pose()
        obs = self._get_observation()
        distance = obs["vector"][0]

        # Check waypoint progress
        goal_bonus = 0.0
        if distance < self.cfg.goal_tolerance:
            self._current_waypoint_idx += 1
            goal_bonus = self.cfg.reward_goal_bonus
            print(f"  Waypoint {self._current_waypoint_idx}/{self.cfg.num_waypoints} reached!")

            if self._current_waypoint_idx < self.cfg.num_waypoints:
                next_wp = self._waypoints[self._current_waypoint_idx]
                self._goal_marker.set_world_pose(
                    position=np.array([next_wp[0], next_wp[1], 0.3])
                )
                # Recalculate observation with new waypoint
                obs = self._get_observation()
                distance = obs["vector"][0]

        # Check termination
        terminated, truncated, _ = self._check_termination(position, orientation)

        # All waypoints completed
        if self._current_waypoint_idx >= self.cfg.num_waypoints:
            terminated = True
            goal_bonus += self.cfg.reward_goal_bonus * 2  # Extra bonus for completing all
            print("  All waypoints completed!")

        # Calculate reward
        reward = self._calculate_reward(distance, terminated)
        reward += goal_bonus

        # Update previous distance
        self._prev_distance = distance

        return obs, reward, terminated, truncated, {}

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode == "rgb_array" and self._camera is not None:
            return self._get_observation().get("image")
        return None

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "_sim_app") and self._sim_app is not None:
            self._sim_app.close()

    @property
    def sim_app(self):
        """Access to SimulationApp for external control."""
        return self._sim_app


# Register with Gymnasium
def register_env():
    """Register Leatherback environment with Gymnasium."""
    gym.register(
        id="Leatherback-v0",
        entry_point="isaac_lab.envs.leatherback_env:LeatherbackEnv",
        max_episode_steps=10000,
    )


if __name__ == "__main__":
    """Test the environment."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Leatherback Environment")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=500, help="Number of test steps")
    args = parser.parse_args()

    print("=" * 60)
    print("Leatherback Environment Test")
    print("=" * 60)

    cfg = LeatherbackEnvCfg(use_camera=False, use_lidar=False)
    env = LeatherbackEnv(cfg=cfg, headless=args.headless)

    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Vector obs shape: {obs['vector'].shape}")

    print(f"\nRunning {args.steps} steps...")
    total_reward = 0.0

    try:
        for i in range(args.steps):
            if not env.sim_app.is_running():
                break

            # Simple forward + oscillating steering
            action = np.array([0.8, np.sin(i * 0.05) * 0.3], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if i % 50 == 0:
                print(f"Step {i}: distance={obs['vector'][0]:.2f}m, reward={reward:.2f}")

            if terminated or truncated:
                print(f"Episode ended at step {i}")
                obs, info = env.reset()
                total_reward = 0.0

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    print(f"\nTotal reward: {total_reward:.2f}")
    env.close()
    print("Test complete!")

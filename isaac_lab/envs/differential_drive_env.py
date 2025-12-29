"""
4-Wheel Differential Drive (Skid-Steer) Robot Environment for Isaac Sim 5.1.0.

Uses the Clearpath Jackal robot model from Isaac Sim assets.
Implements Gymnasium-compatible RL environment for autonomous navigation.

Supported robots:
- Clearpath Jackal (default): 4-wheel skid-steer, 2 cameras, IMU
- Clearpath Dingo: 2-wheel differential
- AgileX Limo: 4-wheel differential

Key features:
- Uses actual robot USD models from Isaac Sim assets
- DifferentialController with skid-steer correction factor
- Same sensor support as Leatherback: Camera + LiDAR + Vector
"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .differential_drive_env_cfg import DifferentialDriveEnvCfg
except ImportError:
    from differential_drive_env_cfg import DifferentialDriveEnvCfg


# Robot USD paths relative to Isaac assets root
ROBOT_PATHS = {
    "jackal": "Isaac/Robots/Clearpath/Jackal/jackal.usd",
    "dingo": "Isaac/Robots/Clearpath/Dingo/dingo.usd",
    "limo": "Isaac/Robots/AgilexRobotics/limo/limo.usd",
}

# Robot-specific configurations
ROBOT_CONFIGS = {
    "jackal": {
        "wheel_radius": 0.098,
        "track_width": 0.37558,
        "chassis_prim": "base_link",
        "wheel_joints": [
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
    },
    "dingo": {
        "wheel_radius": 0.049,
        "track_width": 0.38,
        "chassis_prim": "base_link",
        "wheel_joints": ["front_left_wheel_joint", "front_right_wheel_joint"],
    },
    "limo": {
        "wheel_radius": 0.045,
        "track_width": 0.172,
        "chassis_prim": "base_link",
        "wheel_joints": [
            "front_left_wheel_joint",
            "front_right_wheel_joint",
            "rear_left_wheel_joint",
            "rear_right_wheel_joint",
        ],
    },
}


class DifferentialDriveEnv(gym.Env):
    """4-Wheel Differential Drive robot navigation environment.

    Uses Clearpath Jackal or other differential drive robots from Isaac Sim.
    Compatible with Stable-Baselines3 for RL training.

    Args:
        cfg: Environment configuration.
        headless: Run without GUI.
        render_mode: Gymnasium render mode.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: DifferentialDriveEnvCfg | None = None,
        headless: bool = False,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.cfg = cfg if cfg is not None else DifferentialDriveEnvCfg()
        self.render_mode = render_mode
        self._headless = headless

        print(f"Initializing Differential Drive environment (headless={headless})...")
        print(f"  Robot type: {self.cfg.robot_type}")

        # Initialize Isaac Sim
        from isaacsim import SimulationApp
        self._sim_app = SimulationApp({"headless": headless})

        # Import Isaac modules
        self._import_isaac_modules()

        # Setup simulation
        self._setup_world()
        self._load_robot()
        self._configure_physics()
        self._setup_sensors()
        self._spawn_obstacles()
        self._create_goal_marker()

        # Initialize world
        self._world.reset()

        # Get joint indices and setup controller
        self._get_joint_indices()
        self._setup_differential_controller()

        # Define spaces
        self._define_spaces()

        # State variables
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._current_action = np.zeros(2, dtype=np.float32)
        self._prev_distance = 0.0
        self._prev_position: np.ndarray | None = None
        self._stuck_counter = 0
        self._waypoints: np.ndarray | None = None
        self._current_waypoint_idx = 0

        print("Differential Drive environment initialized successfully!")

    def _import_isaac_modules(self) -> None:
        """Import Isaac Sim modules after SimulationApp initialization."""
        from isaacsim.core.api import World
        from isaacsim.storage.native import get_assets_root_path
        import isaacsim.core.utils.stage as stage_utils
        import isaacsim.core.utils.prims as prim_utils
        import isaacsim.core.utils.xforms as xforms_utils
        from isaacsim.core.utils.types import ArticulationAction

        from pxr import UsdLux, UsdPhysics, PhysxSchema

        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.objects import VisualSphere, FixedCuboid

        # Differential controller for wheeled robots
        from isaacsim.robot.wheeled_robots.controllers.differential_controller import (
            DifferentialController,
        )

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
        self._DifferentialController = DifferentialController
        self._ArticulationAction = ArticulationAction

    def _setup_world(self) -> None:
        """Create simulation world."""
        self._world = self._World()
        self._world.scene.add_default_ground_plane()
        self._stage = self._world.stage

        # Lighting
        dome_light = self._UsdLux.DomeLight.Define(self._stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(2000.0)
        print("  World created")

    def _load_robot(self) -> None:
        """Load robot from Isaac Sim assets."""
        robot_prim_path = "/World/Robot"

        # Determine USD path
        if self.cfg.robot_usd_path is not None:
            robot_usd_path = self.cfg.robot_usd_path
        else:
            assets_root = self._get_assets_root_path()
            if assets_root is None:
                raise RuntimeError(
                    "Isaac Sim assets not configured! "
                    "Run Isaac Sim once to download assets or set ISAAC_NUCLEUS_DIR."
                )
            robot_rel_path = ROBOT_PATHS.get(self.cfg.robot_type, ROBOT_PATHS["jackal"])
            robot_usd_path = f"{assets_root}/{robot_rel_path}"

        # Add robot to stage
        self._stage_utils.add_reference_to_stage(robot_usd_path, robot_prim_path)
        print(f"  Robot loaded from: {robot_usd_path}")

        # Create Articulation wrapper
        self._robot = self._Articulation(
            prim_path=robot_prim_path,
            name="diff_drive_robot",
        )
        self._world.scene.add(self._robot)

        # Store paths
        self._robot_prim_path = robot_prim_path

        # Get chassis path for sensors
        robot_config = ROBOT_CONFIGS.get(self.cfg.robot_type, ROBOT_CONFIGS["jackal"])
        self._chassis_prim_name = robot_config.get("chassis_prim", self.cfg.chassis_prim_name)
        self._chassis_path = f"{robot_prim_path}/{self._chassis_prim_name}"

    def _configure_physics(self) -> None:
        """Configure physics parameters."""
        # Disable self-collisions
        robot_prim = self._stage.GetPrimAtPath(self._robot_prim_path)
        if robot_prim.IsValid():
            physx_articulation = self._PhysxSchema.PhysxArticulationAPI.Apply(robot_prim)
            physx_articulation.GetEnabledSelfCollisionsAttr().Set(False)

        # Configure solver
        try:
            self._world.get_physics_context().set_solver_type(self.cfg.solver_type)
        except AttributeError:
            pass

        # Configure wheel drives
        self._configure_wheel_drives()

        print(f"  Physics configured: {self.cfg.solver_type}")

    def _configure_wheel_drives(self) -> None:
        """Configure drive parameters for wheel joints."""
        robot_config = ROBOT_CONFIGS.get(self.cfg.robot_type, ROBOT_CONFIGS["jackal"])
        wheel_joints = robot_config.get("wheel_joints", list(self.cfg.wheel_joint_names))

        for joint_name in wheel_joints:
            # Try different path patterns
            possible_paths = [
                f"{self._robot_prim_path}/{joint_name}",
                f"{self._robot_prim_path}/Joints/{joint_name}",
                f"{self._chassis_path}/{joint_name}",
            ]

            for prim_path in possible_paths:
                prim = self._stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    drive = self._UsdPhysics.DriveAPI.Apply(prim, "angular")
                    drive.GetStiffnessAttr().Set(self.cfg.wheel_stiffness)
                    drive.GetDampingAttr().Set(self.cfg.wheel_damping)
                    break

    def _get_joint_indices(self) -> None:
        """Get DOF indices for wheel joints."""
        robot_config = ROBOT_CONFIGS.get(self.cfg.robot_type, ROBOT_CONFIGS["jackal"])
        wheel_joints = robot_config.get("wheel_joints", list(self.cfg.wheel_joint_names))

        self._wheel_indices = []
        self._is_4_wheel = len(wheel_joints) >= 4

        # Try to get indices by name
        for joint_name in wheel_joints:
            try:
                idx = self._robot.get_dof_index(joint_name)
                self._wheel_indices.append(idx)
            except Exception:
                pass

        # Fallback if name lookup fails
        if len(self._wheel_indices) < 2:
            num_dofs = self._robot.num_dof
            self._wheel_indices = list(range(min(4, num_dofs)))
            print(f"  Warning: Using fallback wheel indices: {self._wheel_indices}")
        else:
            print(f"  Wheel joint indices: {self._wheel_indices}")

    def _setup_differential_controller(self) -> None:
        """Setup differential drive controller with skid-steer correction."""
        # For skid-steer 4-wheel robots, scale the wheel_base parameter
        effective_wheel_distance = self.cfg.track_width * self.cfg.skid_steer_correction

        self._diff_controller = self._DifferentialController(
            name="diff_controller",
            wheel_radius=self.cfg.wheel_radius,
            wheel_base=effective_wheel_distance,
        )

        print(f"  Differential controller initialized")
        print(f"    Wheel radius: {self.cfg.wheel_radius}m")
        print(f"    Effective wheel distance: {effective_wheel_distance:.2f}m")
        print(f"    (actual: {self.cfg.track_width}m x {self.cfg.skid_steer_correction} correction)")

    def _setup_sensors(self) -> None:
        """Setup camera and LiDAR sensors."""
        self._camera = None
        self._lidar = None

        if self.cfg.use_camera:
            try:
                from omni.isaac.sensor import Camera

                # Try to find built-in camera first (Jackal has 2 cameras)
                camera_paths = [
                    f"{self._robot_prim_path}/front_camera",
                    f"{self._robot_prim_path}/navsat_link/front_camera",
                    f"{self._chassis_path}/front_camera",
                ]

                camera_prim_path = None
                for path in camera_paths:
                    prim = self._stage.GetPrimAtPath(path)
                    if prim.IsValid():
                        camera_prim_path = path
                        print(f"  Found built-in camera at: {path}")
                        break

                if camera_prim_path:
                    self._camera = Camera(
                        prim_path=camera_prim_path,
                        name="front_camera",
                        frequency=30,
                        resolution=self.cfg.camera_resolution,
                    )
                else:
                    # Create new camera
                    camera_prim_path = f"{self._chassis_path}/FrontCamera"
                    print(f"  Creating camera at: {camera_prim_path}")
                    self._camera = Camera(
                        prim_path=camera_prim_path,
                        name="front_camera",
                        position=np.array(self.cfg.camera_position),
                        frequency=30,
                        resolution=self.cfg.camera_resolution,
                        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    )

                self._world.scene.add(self._camera)
                # CRITICAL: Initialize camera for it to work
                self._camera.initialize()
                self._camera.add_rgb_to_frame()
                print(f"  Camera: {self.cfg.camera_resolution} - initialized")
            except Exception as e:
                print(f"  Warning: Camera init failed: {e}")
                import traceback
                traceback.print_exc()

        if self.cfg.use_lidar:
            self._lidar_annotator = None
            self._lidar_render_product = None
            try:
                import omni.kit.commands
                from pxr import Gf
                import omni.replicator.core as rep

                lidar_path = f"{self._chassis_path}/TopLidar"

                # Create RTX LiDAR using official Isaac Sim command
                # Using Example_Rotary_2D for 2D horizontal scan (single plane, 360°)
                # This is required for FlatScan annotator which only works with 2D LiDAR
                _, self._lidar_prim = omni.kit.commands.execute(
                    "IsaacSensorCreateRtxLidar",
                    path=lidar_path,
                    parent=None,
                    config="Example_Rotary_2D",  # 2D 360° rotating LiDAR (horizontal plane only)
                    translation=tuple(self.cfg.lidar_position),
                    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),
                )

                # Create render product (required for annotators)
                self._lidar_render_product = rep.create.render_product(
                    self._lidar_prim.GetPath(), [1, 1], name="LidarRenderProduct"
                )

                # Try FlatScan annotator first (provides per-tick data, more immediate)
                # This is better for 2D navigation use cases
                # NOTE: Isaac Sim 5.1 uses simpler annotator names without "RtxSensorCpu" prefix
                try:
                    self._lidar_annotator = rep.AnnotatorRegistry.get_annotator(
                        "IsaacComputeRTXLidarFlatScan"
                    )
                    self._lidar_annotator.attach(self._lidar_render_product)
                    self._lidar_annotator_type = "FlatScan"
                    print(f"  LiDAR: RTX LiDAR with FlatScan annotator (per-tick)")
                except Exception as e1:
                    print(f"  FlatScan annotator failed: {e1}, trying ScanBuffer...")
                    # Fallback to ScanBuffer (waits for full rotation)
                    self._lidar_annotator = rep.AnnotatorRegistry.get_annotator(
                        "IsaacCreateRTXLidarScanBuffer"
                    )
                    self._lidar_annotator.attach(self._lidar_render_product)
                    self._lidar_annotator_type = "ScanBuffer"
                    print(f"  LiDAR: RTX LiDAR with ScanBuffer annotator (full rotation)")

                print(f"  Config: Example_Rotary_2D (360° horizontal plane)")

            except Exception as e:
                print(f"  Warning: LiDAR init failed: {e}")
                import traceback
                traceback.print_exc()
                self._lidar_annotator = None
                self._lidar_render_product = None

    def _spawn_obstacles(self) -> None:
        """Spawn BARN-style obstacle course.

        Supports multiple course types:
        - "random": Random cylindrical obstacles
        - "corridor": Obstacles forming corridors
        - "maze": Dense maze-like pattern
        - "barn": BARN benchmark style (cellular automata inspired)
        """
        self._obstacles = []
        self._obstacle_positions = []  # Track positions for collision checking
        self._episode_count = getattr(self, '_episode_count', 0)

        # Get current difficulty (curriculum or fixed)
        difficulty = self._get_current_difficulty()

        # Calculate number of obstacles based on difficulty
        num_obstacles = int(
            self.cfg.num_obstacles_min +
            difficulty * (self.cfg.num_obstacles_max - self.cfg.num_obstacles_min)
        )

        # Get course type
        course_type = getattr(self.cfg, 'obstacle_course_type', 'random')

        try:
            if course_type == "barn":
                self._spawn_barn_obstacles(num_obstacles, difficulty)
            elif course_type == "corridor":
                self._spawn_corridor_obstacles(num_obstacles, difficulty)
            elif course_type == "maze":
                self._spawn_maze_obstacles(num_obstacles, difficulty)
            else:
                self._spawn_random_obstacles(num_obstacles)
        except Exception as e:
            print(f"  [ERROR] Obstacle spawning failed: {e}")
            import traceback
            traceback.print_exc()

        print(f"  {len(self._obstacles)} obstacles spawned (difficulty={difficulty:.2f})")

    def _get_current_difficulty(self) -> float:
        """Get current difficulty level (supports curriculum learning)."""
        if not getattr(self.cfg, 'use_curriculum', False):
            return getattr(self.cfg, 'obstacle_difficulty', 0.5)

        # Curriculum: linearly increase difficulty over episodes
        start = getattr(self.cfg, 'curriculum_start_difficulty', 0.2)
        end = getattr(self.cfg, 'curriculum_end_difficulty', 0.9)
        episodes_to_max = getattr(self.cfg, 'curriculum_episodes_to_max', 500)

        progress = min(1.0, self._episode_count / episodes_to_max)
        return start + progress * (end - start)

    def _get_obstacle_spawn_limit(self) -> float:
        """Get the maximum X position for obstacle spawning.

        Obstacles should spawn BEFORE the furthest goal, not behind it.
        This ensures all obstacles are relevant for navigation.
        """
        # Use config value - don't depend on progressive goals during init
        # because waypoints aren't generated until reset()
        spawn_limit = getattr(self.cfg, 'obstacle_spawn_x_max', 18.0)
        return spawn_limit

    def _spawn_barn_obstacles(self, num_obstacles: int, difficulty: float) -> None:
        """Spawn BARN-style obstacles using cellular automata.

        Based on the BARN Challenge methodology:
        https://www.cs.utexas.edu/~xiao/BARN/BARN.html

        Uses cellular automata to generate realistic cluttered environments
        with guaranteed passages (like the actual BARN benchmark).

        IMPORTANT: Obstacles spawn ONLY between robot and goals, not behind goals.
        """
        course_width = getattr(self.cfg, 'course_width', 8.0)
        spawn_x_min = getattr(self.cfg, 'obstacle_spawn_x_min', 2.0)

        # Determine spawn_x_max based on furthest goal (if progressive goals enabled)
        spawn_x_max = self._get_obstacle_spawn_limit()

        # Generate occupancy grid using cellular automata
        grid, cell_size = self._generate_cellular_automata_grid(
            spawn_x_min, spawn_x_max, course_width, difficulty
        )

        # Convert grid cells to obstacle positions
        obstacle_idx = 0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i, j] == 1 and obstacle_idx < num_obstacles:
                    # Convert grid coords to world coords
                    x = spawn_x_min + (i + 0.5) * cell_size
                    y = -course_width / 2 + (j + 0.5) * cell_size
                    pos = np.array([x, y])

                    # Safety check: not too close to start
                    if np.linalg.norm(pos) >= self.cfg.obstacle_min_spawn_distance:
                        self._obstacle_positions.append(pos)
                        self._spawn_single_obstacle_at(obstacle_idx, pos)
                        obstacle_idx += 1

    def _generate_cellular_automata_grid(
        self, x_min: float, x_max: float, width: float, difficulty: float
    ) -> tuple[np.ndarray, float]:
        """Generate obstacle grid using cellular automata (BARN-style).

        Based on procedural generation techniques used in BARN benchmark.

        Returns:
            grid: 2D numpy array (1 = obstacle, 0 = free)
            cell_size: Size of each cell in meters
        """
        # Grid resolution (smaller cells = more detail)
        cell_size = 0.5  # 50cm cells

        # Grid dimensions
        nx = int((x_max - x_min) / cell_size)
        ny = int(width / cell_size)

        # Initialize random grid based on difficulty
        # Higher difficulty = more initial obstacles
        fill_prob = 0.35 + 0.25 * difficulty  # 35-60% filled initially
        grid = (np.random.random((nx, ny)) < fill_prob).astype(np.int32)

        # Clear start zone (robot spawn area)
        start_clear = int(1.5 / cell_size)  # Clear 1.5m
        grid[:start_clear, :] = 0

        # Clear goal zone
        goal_clear = int(1.0 / cell_size)
        grid[-goal_clear:, :] = 0

        # Clear center path (ensure navigability)
        center_y = ny // 2
        path_width = max(2, int((1.0 - 0.5 * difficulty) * 3))  # Narrower at higher difficulty
        grid[:, center_y - path_width:center_y + path_width] = 0

        # Apply cellular automata rules (cave generation style)
        # This creates more natural-looking obstacle clusters
        for _ in range(3):  # 3 iterations
            grid = self._cellular_automata_step(grid, difficulty)

        # Ensure start and goal are still clear
        grid[:start_clear, :] = 0
        grid[-goal_clear:, :] = 0

        # Guarantee a navigable path exists using flood fill
        grid = self._ensure_path_exists(grid, center_y)

        return grid, cell_size

    def _cellular_automata_step(self, grid: np.ndarray, difficulty: float) -> np.ndarray:
        """Apply one step of cellular automata rules.

        Uses a variant of the 4-5 rule:
        - Cell becomes obstacle if >= birth_threshold neighbors are obstacles
        - Cell stays obstacle if >= survival_threshold neighbors are obstacles
        """
        nx, ny = grid.shape
        new_grid = grid.copy()

        # Thresholds based on difficulty
        birth_threshold = 5 - int(difficulty * 2)      # 3-5 (easier birth at high diff)
        survival_threshold = 4 - int(difficulty * 1)   # 3-4

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                # Count neighbors (8-connected)
                neighbors = np.sum(grid[i-1:i+2, j-1:j+2]) - grid[i, j]

                if grid[i, j] == 0:
                    # Birth rule
                    if neighbors >= birth_threshold:
                        new_grid[i, j] = 1
                else:
                    # Survival rule
                    if neighbors < survival_threshold:
                        new_grid[i, j] = 0

        return new_grid

    def _ensure_path_exists(self, grid: np.ndarray, center_y: int) -> np.ndarray:
        """Ensure a navigable path exists from start to goal.

        Uses a simple carving approach if no path found.
        """
        nx, ny = grid.shape

        # Simple path check: can we get from start to end?
        # Use a wavefront/flood fill from start
        visited = np.zeros_like(grid, dtype=bool)
        start_cells = [(0, j) for j in range(ny) if grid[0, j] == 0]

        if not start_cells:
            # No free cells at start - clear center
            grid[0, center_y-1:center_y+2] = 0
            start_cells = [(0, center_y)]

        queue = list(start_cells)
        for cell in queue:
            visited[cell] = True

        reached_goal = False
        while queue and not reached_goal:
            i, j = queue.pop(0)
            if i >= nx - 1:
                reached_goal = True
                break

            # Check 4-connected neighbors
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    if not visited[ni, nj] and grid[ni, nj] == 0:
                        visited[ni, nj] = True
                        queue.append((ni, nj))

        if not reached_goal:
            # Carve a simple path through
            y = center_y
            for x in range(nx):
                # Clear current position and neighbors
                for dy in [-1, 0, 1]:
                    if 0 <= y + dy < ny:
                        grid[x, y + dy] = 0

                # Occasionally shift y for variation
                if np.random.random() < 0.3:
                    y = max(1, min(ny - 2, y + np.random.randint(-1, 2)))

        return grid

    def _generate_barn_position(
        self, x_min: float, x_max: float, width: float, difficulty: float
    ) -> np.ndarray | None:
        """Generate a valid obstacle position for BARN-style course."""
        min_spacing = getattr(self.cfg, 'min_passage_width', 0.8)

        for _ in range(50):
            # Random position in the course area (forward direction)
            x = np.random.uniform(x_min, x_max)
            y = np.random.uniform(-width / 2, width / 2)
            pos = np.array([x, y])

            # Check distance from robot spawn (0, 0)
            if np.linalg.norm(pos) < self.cfg.obstacle_min_spawn_distance:
                continue

            # Check distance from other obstacles (ensure passages exist)
            valid = True
            for other_pos in self._obstacle_positions:
                dist = np.linalg.norm(pos - other_pos)
                # At higher difficulty, allow closer spacing
                min_dist = min_spacing * (1.5 - 0.5 * difficulty)
                if dist < min_dist:
                    valid = False
                    break

            if valid:
                self._obstacle_positions.append(pos)
                return pos

        return None

    def _spawn_corridor_obstacles(self, num_obstacles: int, difficulty: float) -> None:
        """Spawn obstacles forming corridor patterns."""
        course_width = getattr(self.cfg, 'course_width', 8.0)
        spawn_x_min = getattr(self.cfg, 'obstacle_spawn_x_min', 2.0)
        spawn_x_max = getattr(self.cfg, 'obstacle_spawn_x_max', 18.0)

        # Create corridor walls with gaps
        num_walls = int(3 + difficulty * 4)  # 3-7 walls based on difficulty
        wall_spacing = (spawn_x_max - spawn_x_min) / num_walls

        obstacle_idx = 0
        for wall_i in range(num_walls):
            wall_x = spawn_x_min + wall_i * wall_spacing

            # Gap position (random)
            gap_y = np.random.uniform(-course_width / 3, course_width / 3)
            gap_width = getattr(self.cfg, 'max_passage_width', 2.0) * (1.0 - 0.5 * difficulty)

            # Place obstacles on both sides of gap
            y_positions = []
            y = -course_width / 2
            while y < course_width / 2:
                if abs(y - gap_y) > gap_width / 2:
                    y_positions.append(y)
                y += 0.6  # Obstacle spacing

            for y in y_positions:
                if obstacle_idx >= num_obstacles:
                    break
                pos = np.array([wall_x, y])
                self._obstacle_positions.append(pos)
                self._spawn_single_obstacle_at(obstacle_idx, pos)
                obstacle_idx += 1

    def _spawn_maze_obstacles(self, num_obstacles: int, difficulty: float) -> None:
        """Spawn dense maze-like obstacle pattern."""
        # Grid-based placement with random removal for paths
        course_width = getattr(self.cfg, 'course_width', 8.0)
        spawn_x_min = getattr(self.cfg, 'obstacle_spawn_x_min', 2.0)
        spawn_x_max = getattr(self.cfg, 'obstacle_spawn_x_max', 18.0)

        grid_spacing = 1.0 + (1.0 - difficulty) * 0.5  # Tighter grid at higher difficulty
        removal_prob = 0.4 + (1.0 - difficulty) * 0.3  # More removed at lower difficulty

        obstacle_idx = 0
        x = spawn_x_min
        while x < spawn_x_max and obstacle_idx < num_obstacles:
            y = -course_width / 2
            while y < course_width / 2 and obstacle_idx < num_obstacles:
                # Randomly skip some positions to create paths
                if np.random.random() > removal_prob:
                    pos = np.array([x, y])
                    if np.linalg.norm(pos) >= self.cfg.obstacle_min_spawn_distance:
                        self._obstacle_positions.append(pos)
                        self._spawn_single_obstacle_at(obstacle_idx, pos)
                        obstacle_idx += 1
                y += grid_spacing
            x += grid_spacing

    def _spawn_random_obstacles(self, num_obstacles: int) -> None:
        """Spawn random obstacles (original behavior)."""
        for i in range(num_obstacles):
            pos = self._generate_random_position()
            if pos is not None:
                self._spawn_single_obstacle_at(i, pos)

    def _generate_random_position(self) -> np.ndarray | None:
        """Generate a random obstacle position in the forward direction."""
        course_width = getattr(self.cfg, 'course_width', 8.0)
        spawn_x_min = getattr(self.cfg, 'obstacle_spawn_x_min', 2.0)
        spawn_x_max = getattr(self.cfg, 'obstacle_spawn_x_max', 18.0)

        for _ in range(50):
            x = np.random.uniform(spawn_x_min, spawn_x_max)
            y = np.random.uniform(-course_width / 2, course_width / 2)
            pos = np.array([x, y])

            if np.linalg.norm(pos) >= self.cfg.obstacle_min_spawn_distance:
                self._obstacle_positions.append(pos)
                return pos
        return None

    def _spawn_single_obstacle_at(self, index: int, pos: np.ndarray) -> None:
        """Spawn a single obstacle at the given position."""
        obstacle_shape = getattr(self.cfg, 'obstacle_shape', 'cylinder')

        # Randomize shape if mixed
        if obstacle_shape == "mixed":
            obstacle_shape = np.random.choice(["cylinder", "cube"])

        # Get color
        if self.cfg.randomize_obstacle_colors:
            color = np.array([
                np.random.uniform(0.4, 0.9),
                np.random.uniform(0.2, 0.6),
                np.random.uniform(0.1, 0.4),
            ])
        else:
            color = np.array([0.7, 0.3, 0.2])

        if obstacle_shape == "cylinder":
            self._spawn_cylinder_obstacle(index, pos, color)
        else:
            self._spawn_cube_obstacle(index, pos, color)

    def _spawn_cylinder_obstacle(
        self, index: int, pos: np.ndarray, color: np.ndarray
    ) -> None:
        """Spawn a cylindrical obstacle."""
        from omni.isaac.core.objects import DynamicCylinder

        radius = np.random.uniform(
            getattr(self.cfg, 'cylinder_radius_min', 0.15),
            getattr(self.cfg, 'cylinder_radius_max', 0.4)
        )
        height = np.random.uniform(
            getattr(self.cfg, 'cylinder_height_min', 0.5),
            getattr(self.cfg, 'cylinder_height_max', 1.2)
        )

        try:
            # Use FixedCuboid with cylinder-like proportions as fallback
            # (Isaac Sim cylinder creation can be tricky)
            obstacle = self._FixedCuboid(
                prim_path=f"/World/Obstacle_{index}",
                name=f"obstacle_{index}",
                position=np.array([pos[0], pos[1], height / 2]),
                scale=np.array([radius * 2, radius * 2, height]),
                color=color,
            )
            self._world.scene.add(obstacle)
            self._obstacles.append(obstacle)
        except Exception as e:
            print(f"  Warning: Cylinder spawn failed: {e}")

    def _spawn_cube_obstacle(
        self, index: int, pos: np.ndarray, color: np.ndarray
    ) -> None:
        """Spawn a cube obstacle."""
        size = np.array([
            np.random.uniform(self.cfg.obstacle_size_min[0], self.cfg.obstacle_size_max[0]),
            np.random.uniform(self.cfg.obstacle_size_min[1], self.cfg.obstacle_size_max[1]),
            np.random.uniform(self.cfg.obstacle_size_min[2], self.cfg.obstacle_size_max[2]),
        ])

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
        """Re-randomize obstacle positions on reset (BARN-style)."""
        if not self.cfg.randomize_obstacles_on_reset:
            return

        self._episode_count = getattr(self, '_episode_count', 0) + 1
        self._obstacle_positions = []

        difficulty = self._get_current_difficulty()
        course_width = getattr(self.cfg, 'course_width', 8.0)
        spawn_x_min = getattr(self.cfg, 'obstacle_spawn_x_min', 2.0)
        # Use dynamic spawn limit (obstacles ONLY before goals)
        spawn_x_max = self._get_obstacle_spawn_limit()

        for obstacle in self._obstacles:
            pos = self._generate_barn_position(
                spawn_x_min, spawn_x_max, course_width, difficulty
            )
            if pos is None:
                # Fallback to random position
                pos = np.array([
                    np.random.uniform(spawn_x_min, spawn_x_max),
                    np.random.uniform(-course_width / 2, course_width / 2)
                ])
                self._obstacle_positions.append(pos)

            height = np.random.uniform(
                getattr(self.cfg, 'cylinder_height_min', 0.5),
                getattr(self.cfg, 'cylinder_height_max', 1.2)
            )

            try:
                obstacle.set_world_pose(position=np.array([pos[0], pos[1], height / 2]))
            except Exception:
                pass

    def _create_goal_marker(self) -> None:
        """Create visual marker for goal waypoint."""
        self._goal_marker = self._VisualSphere(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            radius=0.3,
            color=np.array([0.0, 1.0, 0.0]),
            position=np.array([100.0, 100.0, 0.3]),
        )
        self._world.scene.add(self._goal_marker)

    def _define_spaces(self) -> None:
        """Define observation and action spaces.

        Supports two modes:
        1. Multi-modal (camera + lidar + goal): For SB3 MultiInputPolicy
           - "camera": (84, 84, 3) RGB image
           - "lidar": (180,) distance readings
           - "goal": (6,) navigation state [dist, sin, cos, progress, lin_vel, ang_vel]

        2. Single-sensor modes: Original behavior with "vector" + optional sensors
        """
        # Check if multi-modal mode (both camera and lidar enabled)
        self._multi_modal = self.cfg.use_camera and self.cfg.use_lidar

        if self._multi_modal:
            # Multi-modal observation space for SB3 MultiInputPolicy
            goal_obs_size = getattr(self.cfg, 'goal_obs_size', 6)

            obs_dict = {
                # RGB Camera image
                "camera": spaces.Box(
                    low=0,
                    high=255,
                    shape=(self.cfg.camera_resolution[0], self.cfg.camera_resolution[1], 3),
                    dtype=np.uint8,
                ),
                # LiDAR scan (normalized)
                "lidar": spaces.Box(
                    low=0.0,
                    high=1.0,  # Normalized by max_range
                    shape=(self.cfg.lidar_num_points,),
                    dtype=np.float32,
                ),
                # Goal/navigation info
                "goal": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(goal_obs_size,),
                    dtype=np.float32,
                ),
            }
        else:
            # Original single-sensor mode
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

        # Action: [linear_velocity_normalized, angular_velocity_normalized]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

    def _generate_waypoints(self) -> np.ndarray:
        """Generate navigation waypoints with progressive curriculum support.

        Progressive curriculum stages:
        - Stage 1: 1 close goal (learn basic navigation)
        - Stage 2: 2 goals (learn sequencing)
        - Stage 3: 3 goals (full course)
        - Stage 4: 3 goals with max lateral randomness
        """
        # Check if progressive goals enabled
        use_progressive = getattr(self.cfg, 'use_progressive_goals', False)

        if use_progressive:
            num_waypoints, spacing, lateral_range = self._get_progressive_goal_params()
        else:
            num_waypoints = self.cfg.num_waypoints
            spacing = self.cfg.waypoint_spacing
            lateral_range = getattr(self.cfg, 'waypoint_lateral_range', 3.0)

        waypoints = []
        for i in range(num_waypoints):
            x = (i + 1) * spacing
            y = np.random.uniform(-lateral_range, lateral_range)
            waypoints.append([x, y])

        return np.array(waypoints, dtype=np.float32)

    def _get_progressive_goal_params(self) -> tuple[int, float, float]:
        """Get goal parameters based on current training stage.

        Returns:
            (num_waypoints, goal_spacing, lateral_range)
        """
        episode = getattr(self, '_episode_count', 0)

        # Stage thresholds
        stage1_end = getattr(self.cfg, 'stage1_episodes', 200)
        stage2_end = getattr(self.cfg, 'stage2_episodes', 400)
        stage3_end = getattr(self.cfg, 'stage3_episodes', 600)

        if episode < stage1_end:
            # Stage 1: 1 close goal
            num_wp = getattr(self.cfg, 'stage1_num_waypoints', 1)
            dist_min = getattr(self.cfg, 'stage1_goal_distance_min', 4.0)
            dist_max = getattr(self.cfg, 'stage1_goal_distance_max', 6.0)
            spacing = np.random.uniform(dist_min, dist_max)
            lateral = getattr(self.cfg, 'stage1_lateral_range', 1.0)

        elif episode < stage2_end:
            # Stage 2: 2 goals
            num_wp = getattr(self.cfg, 'stage2_num_waypoints', 2)
            spacing = getattr(self.cfg, 'stage2_goal_spacing', 5.0)
            lateral = getattr(self.cfg, 'stage2_lateral_range', 1.5)

        elif episode < stage3_end:
            # Stage 3: 3 goals
            num_wp = getattr(self.cfg, 'stage3_num_waypoints', 3)
            spacing = getattr(self.cfg, 'stage3_goal_spacing', 5.0)
            lateral = getattr(self.cfg, 'stage3_lateral_range', 2.0)

        else:
            # Stage 4: Full challenge
            num_wp = getattr(self.cfg, 'stage4_num_waypoints', 3)
            spacing = getattr(self.cfg, 'stage4_goal_spacing', 5.0)
            lateral = getattr(self.cfg, 'stage4_lateral_range', 2.5)

        return num_wp, spacing, lateral

    def _get_robot_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get robot position and orientation."""
        return self._xforms_utils.get_world_pose(prim_path=self._chassis_path)

    def _extract_nav_info(self, obs: dict[str, np.ndarray]) -> tuple[float, float]:
        """Extract navigation info (distance, heading_alignment) from observation.

        Handles both multi-modal and single-sensor observation formats.

        Returns:
            (distance, heading_alignment): Raw distance in meters and cos(heading_error)
        """
        if "goal" in obs:
            # Multi-modal mode: goal = [dist_norm, sin, cos, progress, lin_vel, ang_vel]
            dist_normalized = obs["goal"][0]
            distance = dist_normalized * self.cfg.arena_radius
            heading_alignment = obs["goal"][2]  # cos(heading_error)
        else:
            # Single-sensor mode: vector = [dist, cos, sin, prev_lin, prev_ang]
            distance = obs["vector"][0]
            heading_alignment = obs["vector"][1]  # cos(heading_error)
        return distance, heading_alignment

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Compute current observation.

        In multi-modal mode (camera + lidar), returns:
        - "camera": (84, 84, 3) RGB image
        - "lidar": (180,) normalized distance readings [0, 1]
        - "goal": (6,) navigation state [dist_norm, sin, cos, progress, lin_vel_norm, ang_vel_norm]

        In single-sensor mode, returns original format with "vector" key.
        """
        position, orientation = self._get_robot_pose()

        # Heading from quaternion [w, x, y, z]
        w, x, y, z = orientation
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        # Goal-relative observations
        current_waypoint = self._waypoints[self._current_waypoint_idx]
        goal_vector = current_waypoint - position[:2]
        distance = np.linalg.norm(goal_vector)
        target_heading = np.arctan2(goal_vector[1], goal_vector[0])
        heading_error = np.arctan2(
            np.sin(target_heading - heading), np.cos(target_heading - heading)
        )

        # Multi-modal mode: camera + lidar + goal
        if getattr(self, '_multi_modal', False):
            # Get camera image
            camera_obs = np.zeros((self.cfg.camera_resolution[0], self.cfg.camera_resolution[1], 3), dtype=np.uint8)
            camera_valid = False
            if self._camera is not None:
                try:
                    rgba = self._camera.get_rgba()
                    if rgba is not None and rgba.size > 0:
                        if len(rgba.shape) == 1:
                            expected_size = self.cfg.camera_resolution[0] * self.cfg.camera_resolution[1] * 4
                            if rgba.size == expected_size:
                                rgba = rgba.reshape((self.cfg.camera_resolution[0], self.cfg.camera_resolution[1], 4))
                        if len(rgba.shape) == 3 and rgba.shape[2] >= 3:
                            camera_obs = rgba[:, :, :3].astype(np.uint8)
                            camera_valid = np.mean(camera_obs) > 0  # Check not all zeros
                except Exception as e:
                    if self._step_count < 5:
                        print(f"  Camera error: {e}")


            # Get LiDAR data (normalized to [0, 1])
            lidar_obs = np.ones(self.cfg.lidar_num_points, dtype=np.float32)  # Default to max range
            lidar_valid = False

            # Use annotator-based approach (official Isaac Sim 5.x method)
            if self._lidar_annotator is not None:
                try:
                    # Get data from annotator (official approach)
                    lidar_data = self._lidar_annotator.get_data()


                    # Extract distance/range data from annotator output
                    raw_data = None
                    if isinstance(lidar_data, dict):
                        # FlatScan uses 'linearDepthData', ScanBuffer uses 'distance'
                        # Order matters: try FlatScan keys first, then ScanBuffer
                        for key in ["linearDepthData", "distance", "range", "data", "buffer"]:
                            if key in lidar_data:
                                data = lidar_data[key]
                                if data is not None and hasattr(data, 'size') and data.size > 0:
                                    raw_data = np.asarray(data).flatten()
                                    # Filter out invalid readings (0 or negative)
                                    valid_mask = raw_data > 0.01
                                    if np.any(valid_mask):
                                        break
                                    else:
                                        # All invalid, try next key
                                        raw_data = None

                    elif hasattr(lidar_data, 'size') and lidar_data.size > 0:
                        raw_data = np.asarray(lidar_data).flatten()

                    if raw_data is not None and raw_data.size > 0:
                        # Replace invalid readings (<=0) with max range
                        raw_data = np.where(raw_data > 0.01, raw_data, self.cfg.lidar_max_range)

                        # Resample to desired number of points
                        if raw_data.size != self.cfg.lidar_num_points:
                            indices = np.linspace(0, raw_data.size - 1, self.cfg.lidar_num_points, dtype=int)
                            lidar_data_resampled = raw_data[indices]
                        else:
                            lidar_data_resampled = raw_data
                        # Normalize to [0, 1]
                        lidar_obs = np.clip(lidar_data_resampled / self.cfg.lidar_max_range, 0.0, 1.0).astype(np.float32)
                        lidar_valid = np.mean(lidar_obs) < 0.99  # Check not all at max range

                except Exception as e:
                    if self._step_count < 5:
                        print(f"  LiDAR annotator error: {e}")
                        import traceback
                        traceback.print_exc()

            # Goal observation: [dist_norm, sin, cos, progress, lin_vel_norm, ang_vel_norm]
            max_dist = getattr(self.cfg, 'arena_radius', 25.0)
            num_waypoints = len(self._waypoints)
            progress = self._current_waypoint_idx / max(num_waypoints, 1)

            goal_obs = np.array([
                np.clip(distance / max_dist, 0.0, 1.0),      # Normalized distance
                np.sin(heading_error),                       # Sin of heading error
                np.cos(heading_error),                       # Cos of heading error
                progress,                                    # Waypoint progress [0, 1]
                self._prev_action[0],                        # Previous linear vel (already [-1, 1])
                self._prev_action[1],                        # Previous angular vel (already [-1, 1])
            ], dtype=np.float32)

            return {
                "camera": camera_obs,
                "lidar": lidar_obs,
                "goal": goal_obs,
            }

        # Original single-sensor mode
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

        # Camera
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

        # LiDAR (single-sensor mode uses same annotator approach)
        if self.cfg.use_lidar and self._lidar_annotator is not None:
            lidar_data = np.zeros(self.cfg.lidar_num_points, dtype=np.float32)
            try:
                annotator_data = self._lidar_annotator.get_data()
                raw_data = None
                if isinstance(annotator_data, dict):
                    for key in ["distance", "range", "linearDepthData", "data", "buffer"]:
                        if key in annotator_data:
                            data = annotator_data[key]
                            if data is not None and hasattr(data, 'size') and data.size > 0:
                                raw_data = np.asarray(data).flatten()
                                break
                elif hasattr(annotator_data, 'size') and annotator_data.size > 0:
                    raw_data = np.asarray(annotator_data).flatten()

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
        """Apply differential drive action.

        Action: [linear_vel_norm, angular_vel_norm] in range [-1, 1]

        Uses DifferentialController to compute wheel velocities.
        For 4-wheel robots, same velocity is applied to front and rear on each side.
        """
        # Clamp actions
        linear_norm = float(np.clip(action[0], -1.0, 1.0))
        angular_norm = float(np.clip(action[1], -1.0, 1.0))

        # Convert to physical units
        linear_vel = linear_norm * self.cfg.max_linear_velocity
        angular_vel = angular_norm * self.cfg.max_angular_velocity

        # Get wheel velocities from differential controller
        command = [linear_vel, angular_vel]
        controller_action = self._diff_controller.forward(command)

        # Extract wheel velocities (left_wheel, right_wheel)
        wheel_vels = controller_action.joint_velocities

        if wheel_vels is not None and len(wheel_vels) >= 2:
            left_vel = float(wheel_vels[0])
            right_vel = float(wheel_vels[1])
        else:
            # Fallback: compute manually
            L = self.cfg.track_width * self.cfg.skid_steer_correction
            r = self.cfg.wheel_radius
            left_vel = (2 * linear_vel - angular_vel * L) / (2 * r)
            right_vel = (2 * linear_vel + angular_vel * L) / (2 * r)

        # Clamp to max wheel velocity
        max_vel = self.cfg.max_wheel_velocity
        left_vel = np.clip(left_vel, -max_vel, max_vel)
        right_vel = np.clip(right_vel, -max_vel, max_vel)

        # Create velocity array for all DOFs
        num_dofs = self._robot.num_dof
        full_velocities = np.zeros(num_dofs, dtype=np.float32)

        # Apply velocities to wheels
        if self._is_4_wheel and len(self._wheel_indices) >= 4:
            # 4-wheel: [front_left, front_right, rear_left, rear_right]
            full_velocities[self._wheel_indices[0]] = left_vel   # front_left
            full_velocities[self._wheel_indices[1]] = right_vel  # front_right
            full_velocities[self._wheel_indices[2]] = left_vel   # rear_left
            full_velocities[self._wheel_indices[3]] = right_vel  # rear_right
        elif len(self._wheel_indices) >= 2:
            # 2-wheel: [left, right]
            full_velocities[self._wheel_indices[0]] = left_vel
            full_velocities[self._wheel_indices[1]] = right_vel

        # Apply action
        robot_action = self._ArticulationAction(
            joint_velocities=full_velocities,
        )
        self._robot.apply_action(robot_action)

    def _calculate_reward(
        self,
        distance: float,
        current_position: np.ndarray,
        terminated: bool,
        obs: dict[str, np.ndarray]
    ) -> float:
        """
        CONDITIONAL DENSE REWARD SYSTEM - Autonomous driving aligned.

        Key principle: Behavior rewards are GATED on actual progress.
        - Spinning → no progress → no heading/velocity/smooth bonus
        - Wrong direction → bad heading → no velocity bonus
        - Stuck → no movement → stuck penalty applies

        This encourages good driving behavior while preventing exploitation.
        """
        reward = 0.0

        # =====================================================
        # 1. CORE: Progress toward goal (ALWAYS ACTIVE)
        # =====================================================
        progress = self._prev_distance - distance

        if progress > 0:
            reward += progress * self.cfg.reward_progress_scale
        else:
            reward += progress * self.cfg.reward_away_penalty_scale

        # =====================================================
        # 2. CONDITIONAL: Heading bonus (ONLY if making progress)
        # =====================================================
        # Get heading alignment from observation (cos(heading_error))
        _, heading_alignment = self._extract_nav_info(obs)  # cos(heading_error), range [-1, 1]

        if progress > self.cfg.progress_gate:
            # Robot is actually moving toward goal - reward good heading
            # Normalize to [0, 1] range: (alignment + 1) / 2
            heading_bonus = ((heading_alignment + 1.0) / 2.0) * self.cfg.reward_heading_scale
            reward += heading_bonus

        # =====================================================
        # 3. CONDITIONAL: Velocity bonus (ONLY if heading is good)
        # =====================================================
        if heading_alignment > self.cfg.heading_gate:
            # Heading is good (within ~45° of goal) - reward speed
            if self._prev_position is not None:
                actual_speed = np.linalg.norm(current_position[:2] - self._prev_position[:2])
                # Cap at reasonable speed (0.5m per step is quite fast)
                velocity_bonus = min(actual_speed, 0.5) * self.cfg.reward_velocity_scale
                reward += velocity_bonus

        # =====================================================
        # 4. CONDITIONAL: Smooth action bonus (ONLY if making progress)
        # =====================================================
        if progress > self.cfg.progress_gate:
            # Robot is progressing - reward smooth control
            action_diff = np.abs(self._current_action - self._prev_action).mean()
            smoothness = max(0.0, 1.0 - action_diff)  # 1.0 = perfectly smooth
            smooth_bonus = smoothness * self.cfg.reward_smooth_scale
            reward += smooth_bonus

        # =====================================================
        # 5. FORWARD PREFERENCE: Penalize reverse motion
        # =====================================================
        # action[0] is linear velocity: negative = reverse
        if self._current_action[0] < 0:
            reward += self.cfg.reward_reverse_penalty

        # =====================================================
        # 6. SAFETY: LiDAR-based obstacle avoidance (ALWAYS ACTIVE)
        # =====================================================
        if self.cfg.use_lidar and "lidar" in obs:
            lidar_data = obs["lidar"]
            if lidar_data is not None and len(lidar_data) > 0:
                valid = lidar_data[(lidar_data > 0.1) & (lidar_data < self.cfg.lidar_max_range)]
                if len(valid) > 0:
                    min_dist = np.min(valid)
                    if min_dist < self.cfg.reward_obstacle_danger_zone:
                        danger_ratio = 1.0 - (min_dist / self.cfg.reward_obstacle_danger_zone)
                        penalty = danger_ratio * self.cfg.reward_obstacle_penalty_max
                        reward -= penalty

        # =====================================================
        # 7. ANTI-STUCK: Penalize if not moving (ALWAYS ACTIVE)
        # =====================================================
        if self._prev_position is not None:
            actual_movement = np.linalg.norm(current_position[:2] - self._prev_position[:2])
            if actual_movement < self.cfg.stuck_movement_threshold:
                self._stuck_counter = getattr(self, '_stuck_counter', 0) + 1
                if self._stuck_counter > self.cfg.stuck_threshold_steps:
                    reward += self.cfg.reward_stuck_penalty
            else:
                self._stuck_counter = 0

        # =====================================================
        # 8. TERMINATION: Strong penalty for failure
        # =====================================================
        if terminated:
            reward += self.cfg.reward_collision_penalty

        return reward

    def _check_termination(
        self, position: np.ndarray, orientation: np.ndarray
    ) -> tuple[bool, bool, float]:
        """Check termination conditions."""
        terminated = False
        truncated = False
        bonus = 0.0

        # Out of bounds
        if np.linalg.norm(position[:2]) > self.cfg.arena_radius:
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

        # Time limit
        max_steps = int(self.cfg.episode_length_s / self.cfg.physics_dt / self.cfg.decimation)
        if self._step_count >= max_steps:
            truncated = True

        return terminated, truncated, bonus

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

        # Reset controller
        self._diff_controller.reset()

        # Reset physics
        self._world.reset()

        # Randomize obstacles
        self._randomize_obstacles()

        # Generate waypoints
        self._waypoints = self._generate_waypoints()
        self._current_waypoint_idx = 0

        # Update goal marker AFTER world reset
        first_wp = self._waypoints[0]
        self._goal_marker.set_world_pose(
            position=np.array([first_wp[0], first_wp[1], 0.3])
        )

        # Zero velocities
        num_dofs = self._robot.num_dof
        zero_action = self._ArticulationAction(
            joint_velocities=np.zeros(num_dofs, dtype=np.float32),
        )
        self._robot.apply_action(zero_action)

        # Step to apply
        self._world.step(render=not self._headless)

        # Reset state
        self._step_count = 0
        self._prev_action = np.zeros(2, dtype=np.float32)
        self._current_action = np.zeros(2, dtype=np.float32)
        self._stuck_counter = 0

        # Get initial observation and position
        obs = self._get_observation()
        self._prev_distance, _ = self._extract_nav_info(obs)
        position, _ = self._get_robot_pose()
        self._prev_position = position.copy()

        print(f"Reset: First waypoint at ({first_wp[0]:.1f}, {first_wp[1]:.1f})")

        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Execute one environment step."""
        self._step_count += 1
        self._current_action = action.copy()

        # Apply action
        self._apply_action(action)

        # Step physics
        self._world.step(render=not self._headless)

        # Get state
        position, orientation = self._get_robot_pose()
        obs = self._get_observation()
        distance, _ = self._extract_nav_info(obs)

        # Check waypoint progress
        goal_bonus = 0.0
        if distance < self.cfg.goal_tolerance:
            self._current_waypoint_idx += 1
            goal_bonus = self.cfg.reward_waypoint_bonus
            num_waypoints = len(self._waypoints)
            print(f"  Waypoint {self._current_waypoint_idx}/{num_waypoints} reached!")

            if self._current_waypoint_idx < num_waypoints:
                next_wp = self._waypoints[self._current_waypoint_idx]
                self._goal_marker.set_world_pose(
                    position=np.array([next_wp[0], next_wp[1], 0.3])
                )
                obs = self._get_observation()
                distance, _ = self._extract_nav_info(obs)

        # Check termination
        terminated, truncated, _ = self._check_termination(position, orientation)

        # All waypoints completed
        num_waypoints = len(self._waypoints)
        if self._current_waypoint_idx >= num_waypoints:
            terminated = True
            goal_bonus += self.cfg.reward_all_waypoints_bonus
            print("  All waypoints completed!")

        # Calculate reward with robust system (no heading/action rewards)
        reward = self._calculate_reward(distance, position, terminated, obs)
        reward += goal_bonus

        # Update state for next step
        self._prev_distance = distance
        self._prev_position = position.copy()
        self._prev_action = action.copy()

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
        """Access to SimulationApp."""
        return self._sim_app


# Register with Gymnasium
def register_env():
    """Register environment with Gymnasium."""
    gym.register(
        id="DiffDrive-v0",
        entry_point="isaac_lab.envs.differential_drive_env:DifferentialDriveEnv",
        max_episode_steps=10000,
    )


if __name__ == "__main__":
    """Test the environment."""
    import argparse

    parser = argparse.ArgumentParser(description="Test Differential Drive Environment")
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--steps", type=int, default=500, help="Test steps")
    parser.add_argument("--robot", type=str, default="jackal",
                        choices=["jackal", "dingo", "limo"],
                        help="Robot type")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Differential Drive Environment Test ({args.robot})")
    print("=" * 60)

    cfg = DifferentialDriveEnvCfg(
        robot_type=args.robot,
        use_camera=False,
        use_lidar=False,
    )
    env = DifferentialDriveEnv(cfg=cfg, headless=args.headless)

    obs, info = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Vector obs shape: {obs['vector'].shape}")

    print(f"\nRunning {args.steps} steps...")
    total_reward = 0.0

    try:
        for i in range(args.steps):
            if not env.sim_app.is_running():
                break

            # Simple forward + oscillating turn
            action = np.array([0.8, np.sin(i * 0.03) * 0.3], dtype=np.float32)
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

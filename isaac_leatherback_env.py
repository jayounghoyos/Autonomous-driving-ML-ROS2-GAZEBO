#!/usr/bin/env python3
"""
Leatherback RL Environment for Isaac Sim - Simplified Working Version
Adapted from Leatherback Community Tutorial
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from isaacsim import SimulationApp



class LeatherbackEnv(gym.Env):
    """
    Simplified Leatherback environment that actually works with Isaac Sim API
    """
    
    def __init__(self, headless=False, use_camera=False):
        super().__init__()
        
        self.use_camera = use_camera
        print(f"Initializing Leatherback environment (Headless={headless}, Camera={use_camera})...")
        
        # Initialize Isaac Sim
        self.sim_app = SimulationApp({"headless": headless})
        
        # Import after SimulationApp
        from isaacsim.core.api import World
        from isaacsim.storage.native import get_assets_root_path
        import isaacsim.core.utils.stage as stage_utils
        from pxr import UsdLux, UsdPhysics, PhysxSchema
        
        # Store for later use
        self.UsdPhysics = UsdPhysics
        self.PhysxSchema = PhysxSchema
        
        # Create world
        self.world = World()
        self.world.scene.add_default_ground_plane()
        print("✓ World created")
        
        # Add lighting
        self.stage = self.world.stage
        dome_light = UsdLux.DomeLight.Define(self.stage, "/World/DomeLight")
        dome_light.CreateIntensityAttr(2000.0)
        print("✓ Lighting added")
        
        # Load Leatherback from local custom assets (from Leatherback-main)
        # This custom asset has better physics configuration than the raw cloud asset
        leatherback_path = "/home/jayoungh/PersonalPorjects/Leatherback-main/source/Leatherback/Leatherback/tasks/direct/leatherback/custom_assets/leatherback_simple_better.usd"
        stage_utils.add_reference_to_stage(leatherback_path, "/World/Leatherback")
        print(f"✓ Leatherback loaded from {leatherback_path}")
        
        
        # Load Leatherback
        assets_root = get_assets_root_path()
        if assets_root is None:
            raise RuntimeError("Assets not configured!")
        
        leatherback_path = assets_root + "/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
        stage_utils.add_reference_to_stage(leatherback_path, "/World/Leatherback")
        print(f"✓ Leatherback loaded from {leatherback_path}")
        
        # Use Articulation class for robust control
        # Using omni.isaac.core as fallback since isaacsim.core.api structure varies
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.objects import VisualSphere, FixedCuboid
        self.robot = Articulation(prim_path="/World/Leatherback", name="leatherback")
        self.world.scene.add(self.robot)
        
        # Disable self collisions properly using PhysX Schema
        # Argument enabled_self_collisions is not available in Articulation __init__
        prim = self.world.stage.GetPrimAtPath("/World/Leatherback")
        if prim.IsValid():
            physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(prim)
            physx_articulation.GetEnabledSelfCollisionsAttr().Set(False)
            print("✓ Self-collisions disabled for Leatherback via PhysX API")
        
        # Check if we can enable CCD (Continuous Collision Detection)
        # Usually done via PhysX settings or RigidBody API
        # For Articulation, proper collision filter and solver iteration count helps
        print("Increasing visualizer/solver iterations for stability...")
        # Note: In Isaac Sim 5.0, direct modification of physics context might vary.
        # We will try to get the PhysX Scene prim and modify attributes directly if the helper fails.
        # But safest is just to set solver type if supported, or skip advanced tuning if API changed.
        try:
             self.world.get_physics_context().set_solver_type("TGS")
        except:
             pass
             
        # Helper to set iterations might be missing. 
        # We can try to find the PhysicsScene prim
        stage = self.world.stage
        scene_prim = stage.GetPrimAtPath("/World/PhysicsScene")
        if scene_prim.IsValid():
            from pxr import PhysxSchema
            physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            physx_scene.GetSolverTypeAttr().Set("TGS")
            physx_scene.GetTimeStepsPerSecondAttr().Set(60)
        
        # Enable CCD on robot
        self._enable_ccd()
        
        # Configure Drives (Critical for Movement)
        # Using UsdPhysics.DriveAPI to set stiffness/damping directly on the prims
        print("Configuring physics drives...")
        stage = self.stage
        
        # Throttle joints (Velocity config: Stiffness=0, Damping > 0)
        throttle_joints = [
            "Wheel__Knuckle__Front_Left", "Wheel__Knuckle__Front_Right",
            "Wheel__Upright__Rear_Right", "Wheel__Upright__Rear_Left"
        ]
        
        for joint_name in throttle_joints:
            # We must find the joint prim. Usually under /World/Leatherback/Joints/ or similar
            # But Articulation handles the path finding internally.
            # Let's search under the robot prim
            import isaacsim.core.utils.prims as prim_utils
            
            # This is a bit of a guess on the path structure, so we iterate to find
            joint_prim = None
            joints_path = "/World/Leatherback/Joints"
            
            # Construct full path
            prim_path = f"{joints_path}/{joint_name}"
            prim = stage.GetPrimAtPath(prim_path)
            
            if prim.IsValid():
                # Apply Drive API if not present
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(0.0)      # Allow free spinning
                drive.GetDampingAttr().Set(10.0)       # SPEED FIX: Reduced drag
                # Mark as velocity drive? Not strictly needed if stiffness is 0
                print(f"  Configured throttle drive: {joint_name}")
            else:
                print(f"  WARNING: Could not find joint prim {prim_path}")

        # Steering joints (Position config: Stiffness > 0, Damping > 0)
        steering_joints = [
            "Knuckle__Upright__Front_Right", "Knuckle__Upright__Front_Left"
        ]
        
        for joint_name in steering_joints:
            prim_path = f"/World/Leatherback/Joints/{joint_name}"
            prim = stage.GetPrimAtPath(prim_path)
            
            if prim.IsValid():
                drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
                drive.GetStiffnessAttr().Set(10000.0)   # Reduced stiffness for stability
                drive.GetDampingAttr().Set(1000.0)      # Damping
                print(f"  Configured steering drive: {joint_name}")
        
        # Define joint names
        self.throttle_joint_names = [
            "Wheel__Knuckle__Front_Left",
            "Wheel__Knuckle__Front_Right",
            "Wheel__Upright__Rear_Right",
            "Wheel__Upright__Rear_Left"
        ]
        self.steering_joint_names = [
            "Knuckle__Upright__Front_Right",
            "Knuckle__Upright__Front_Left",
        ]
        
        # We need to initialize the simulation to get joint indices
        self.world.reset()
        
        # Get indices
        self.throttle_indices = [self.robot.get_dof_index(name) for name in self.throttle_joint_names]
        self.steering_indices = [self.robot.get_dof_index(name) for name in self.steering_joint_names]
        
        print(f"✓ Found throttle indices: {self.throttle_indices}")
        print(f"✓ Found steering indices: {self.steering_indices}")
        
        # Initialize Waypoint Parameters (Must be before reset)
        self.num_waypoints = 10
        self.current_waypoint_idx = 0
        self.waypoints = None
        self.position_tolerance = 0.5
        
        # Goal Marker (Visual only)
        self.goal_marker = VisualSphere(
            prim_path="/World/GoalMarker",
            name="goal_marker",
            radius=0.3,
            color=np.array([0.0, 1.0, 0.0]), # Green
            position=np.array([100.0, 100.0, 0.0]) # Start hidden/far
        )
        self.world.scene.add(self.goal_marker)
        
        # Spawn Obstacles
        self.num_obstacles = 5
        self._spawn_obstacles()
        
        # Walls removed replaced by Virtual Geofence in step()
        
        # Add RGB Camera
        if self.use_camera:
            from omni.isaac.sensor import Camera
            self.camera = Camera(
                prim_path="/World/Leatherback/Camera",
                name="rgb_camera",
                position=np.array([0.5, 0.0, 1.0]), # Roof/Hood mount
                frequency=20,
                resolution=(64, 64),
                orientation=np.array([1.0, 0.0, 0.0, 0.0])
            )
            self.camera.initialize()
            
            # Setup LiDAR
            from omni.isaac.sensor import LidarRtx
            # Generic Rotary Lidar
            # FIX: Removed config_config_name which caused crash. Using basic params.
            # Generic Rotary Lidar
            # FIX: Removed config_config_name which caused crash. Using basic params.
            # FIX 2: Parent to Chassis so it moves with the robot!
            self.lidar = LidarRtx(
                prim_path="/World/Leatherback/Rigid_Bodies/Chassis/Lidar",
                name="lidar",
                position=np.array([0.0, 0.0, 1.2]), # Roof Top relative to Chassis
                orientation=np.array([1.0, 0.0, 0.0, 0.0]), # Standard Z-up
            )
            self.lidar.initialize()
            # ENABLE DATA STREAM
            self.lidar.add_range_data_to_frame()
            self.lidar.enable_visualization()
            
            # Define observation space
            # We use a Dict space for Multi-Input Policy (Image + Vector + Lidar)
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
                "lidar": spaces.Box(low=0.0, high=100.0, shape=(360,), dtype=np.float32)
            })
        else:
            self.camera = None
            self.lidar = None
            # Standard vector space if no camera/lidar
            # But to keep code simple we'll still use Dict or just array?
            # Let's stick to Dict but image/lidar will be zeros
            self.observation_space = spaces.Dict({
                "image": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "vector": spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32),
                "lidar": spaces.Box(low=0.0, high=100.0, shape=(360,), dtype=np.float32)
            })

        # Define action space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Increase iterations further for high-speed wall impacts

        

                
    def _enable_ccd(self):
        """Enable Continuous Collision Detection on all robot links"""
        # print("Enabling CCD on robot links...")
        # from pxr import UsdPhysics, PhysxSchema, Usd
        # # Traverse all children of the robot
        # robot_prim = self.world.stage.GetPrimAtPath("/World/Leatherback")
        # for prim in Usd.PrimRange(robot_prim):
        #     if prim.IsA(UsdPhysics.RigidBodyAPI):
        #          physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        #          physx_rb.GetEnableCCDAttr().Set(True)
        #          # print(f"  + CCD enabled for {prim.GetName()}")
        pass

            
    def _spawn_obstacles(self):
        """Spawn simple obstacles"""
        from omni.isaac.core.objects import FixedCuboid
        print("Spawning obstacles...")
        for i in range(self.num_obstacles):
            # Random position within arena bounds (approx -10 to 10)
            # Avoid center (0,0) where robot spawns
            while True:
                pos = np.random.uniform(-10.0, 10.0, size=2)
                if np.linalg.norm(pos) > 5.0: # Increased safety radius to 5.0m
                    break
            
            obstacle = FixedCuboid(
                prim_path=f"/World/Obstacle_{i}",
                name=f"obstacle_{i}",
                position=np.array([pos[0], pos[1], 0.5]), # On ground (height=1)
                scale=np.array([1.0, 1.0, 1.0]),
                color=np.array([0.8, 0.2, 0.2]) # Red
            )
            self.world.scene.add(obstacle)
            print(f"  Obstacle {i} at {pos}")
        self.num_waypoints = 10
        self.current_waypoint_idx = 0
        self.waypoints = None

        
        print("✓ Leatherback environment ready!")
    
    def _generate_waypoints(self):
        """Generate waypoints"""
        waypoints = []
        for i in range(self.num_waypoints):
            x = (i + 1) * 5.0  # 5 meters apart
            y = np.random.uniform(-3.0, 3.0)
            waypoints.append([x, y])
        return np.array(waypoints, dtype=np.float32)
    
    def _get_observation(self):
        """Get current observation"""
        # Get robot state from Articulation API/PhysX
        # Note: In Isaac Sim 4.0/5.0 Articulation class handles pose reading
        # But for world pose sometimes it's better to use get_world_pose from utils
        import isaacsim.core.utils.xforms as xforms_utils
        # TRACKING FIX: Target the rigid body 'Chassis' not the container
        position, orientation = xforms_utils.get_world_pose(prim_path="/World/Leatherback/Rigid_Bodies/Chassis")
        
        # Calculate heading
        w, x, y, z = orientation
        heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        
        # Debug output
        if hasattr(self, '_step_count'):
            self._step_count += 1
            if self._step_count % 10 == 0:
                 print(f"  Pos: ({position[0]:.2f}, {position[1]:.2f}) | Heading: {heading:.2f}")
        else:
            self._step_count = 0
        
        # Get current waypoint
        current_waypoint = self.waypoints[self.current_waypoint_idx]
        
        # Calculate distance and heading error
        position_error_vector = current_waypoint - position[:2]
        distance = np.linalg.norm(position_error_vector)
        
        target_heading = np.arctan2(position_error_vector[1], position_error_vector[0])
        # Angle difference normalized to [-pi, pi]
        heading_error = np.arctan2(np.sin(target_heading - heading), np.cos(target_heading - heading))
        
        # Build vector observation
        vector_obs = np.array([
            distance,
            np.cos(heading_error),
            np.sin(heading_error),
            self.prev_action[0],
            self.prev_action[1]
        ], dtype=np.float32)
        
        # Get camera image
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        if self.use_camera:
            # Note: In headless mode strictly, this might return empty if renderer is not enabled correctly.
            rgba = self.camera.get_rgba()
            if rgba is not None and rgba.size > 0:
                try:
                    # Isaac Sim Camera often returns flattened array in some configs
                    if len(rgba.shape) == 1:
                        # Assuming 64x64 resolution as defined in setup
                        # Check if size matches
                        expected_size = 64 * 64 * 4
                        if rgba.size == expected_size:
                            rgba = rgba.reshape((64, 64, 4))
                        else:
                            # Try to infer square shape
                            side = int(np.sqrt(rgba.size / 4))
                            rgba = rgba.reshape((side, side, 4))
                            
                    # Remove Alpha channel and ensure dimensions
                    if len(rgba.shape) == 3 and rgba.shape[2] >= 3:
                        candidate_image = rgba[:, :, :3].astype(np.uint8)
                        
                        # Resize if needed (double check against 64x64 requirement)
                        if candidate_image.shape == (64, 64, 3):
                             image = candidate_image
                        else:
                             # Warning but fallback to zeros to prevent crash
                             # print(f"Warning: Camera image shape mismatch: {candidate_image.shape}, expected (64, 64, 3)")
                             pass
                    else:
                        pass # Dimensions wrong
                        
                except Exception as e:
                    print(f"Warning: Failed to process camera image: {e}. Shape: {rgba.shape}")
        
        # Get LiDAR data
        lidar_data = np.zeros(360, dtype=np.float32)
        if self.lidar:
             try:
                 # Standard Replicator/RtxLidar Pattern:
                 # 1. We called add_range_data_to_frame() in init
                 # 2. We get the full frame dictionary here
                 frame = self.lidar.get_current_frame()
                 
                 # Extract 'range' data if it exists
                 # Keys are usually "range" or "linear_depth" depending on what we added
                 raw_data = None
                 if "range" in frame: # Standard key for add_range_data_to_frame
                     raw_data = frame["range"]
                 elif "depth" in frame:
                     raw_data = frame["depth"]
                     
                 if raw_data is not None and raw_data.size > 0:
                     if raw_data.size != 360:
                         indices = np.linspace(0, raw_data.size - 1, 360, dtype=int)
                         lidar_data = raw_data[indices]
                     else:
                         lidar_data = raw_data
                 
                 # Normalize
                 lidar_data = np.clip(lidar_data, 0.0, 20.0)
                 
             except Exception as e:
                 # Silent fail after initial debug
                 # print(f"Warning: LiDAR Read Failed: {e}")
                 pass 
        
        return {
            "image": image,
            "vector": vector_obs,
            "lidar": lidar_data
        }
    
    def _apply_action(self, throttle, steering):
        """Apply throttle and steering using Articulation API"""
        try:
            # Apply velocity to wheels (throttle)
            # 50.0 is the scale factor
            velocity_targets = np.full(4, throttle).astype(np.float32)
            if self.throttle_indices:
                self.robot.set_joint_velocities(velocity_targets, joint_indices=self.throttle_indices)
                    
        except Exception as e:
            print(f"Warning: Could not apply action: {e}")
    
    def _calculate_reward(self, obs):
        """Calculate reward"""
        # Extract from vector part
        vector = obs["vector"]
        distance = vector[0]
        heading_error = np.arctan2(vector[2], vector[1])
        
        # Progress reward
        progress = self.prev_distance - distance
        progress_reward = progress * 10.0
        
        # Time penalty (Encourage speed)
        time_penalty = -0.05
        
        # Heading reward
        heading_reward = np.exp(-abs(heading_error)) * 0.5
        
        # Goal reached
        goal_bonus = 0.0
        if distance < self.position_tolerance:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < self.num_waypoints:
                # Move marker to next waypoint
                next_wp = self.waypoints[self.current_waypoint_idx]
                self.goal_marker.set_world_pose(position=np.array([next_wp[0], next_wp[1], 0.3]))
                print(f"✓ Waypoint reached! Next: {next_wp}")
            
            goal_bonus = 50.0
            print(f"✓ Waypoint {self.current_waypoint_idx}/{self.num_waypoints} reached!")
        
        self.prev_distance = distance
        
        return progress_reward + heading_reward + goal_bonus + time_penalty
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        # Reset robot to origin using Articulation API
        # Randomize Yaw (Rotation around Z)
        import scipy.spatial.transform as transform
        random_yaw = np.random.uniform(0, 2 * np.pi)
        # Euler to Quaternion (w, x, y, z)
        # For simple rotation around Z: w=cos(theta/2), z=sin(theta/2)
        random_ori = np.array([np.cos(random_yaw/2), 0, 0, np.sin(random_yaw/2)])
        
        # Spawn at 0.05m but with random rotation
        self.robot.set_world_pose(position=np.array([0.0, 0.0, 0.05]), orientation=random_ori)
        
        # Zero out velocities for controlled joints
        all_indices = self.throttle_indices + self.steering_indices
        self.robot.set_joint_velocities(np.zeros(len(all_indices)), joint_indices=all_indices)
        
        # Generate new waypoints
        self.waypoints = self._generate_waypoints()
        self.current_waypoint_idx = 0
        
        # Reset previous state
        self.prev_action = np.zeros(2, dtype=np.float32)
        
        # Get observation
        obs = self._get_observation()
        self.prev_distance = obs["vector"][0]
        
        print(f"Environment reset. First waypoint at {self.waypoints[0]}")
        
        # Update goal marker
        start_wp = self.waypoints[0]
        self.goal_marker.set_world_pose(position=np.array([start_wp[0], start_wp[1], 0.3]))
        
        return obs, {}
    
    def step(self, action):
        """Execute step"""
        # Store action
        self.prev_action = action
        
        # Scale actions
        # SPEED FIX: Increased torque limits (Kept as requested)
        throttle = np.clip(action[0], -1.0, 1.0) * 100.0
        # STEERING REVERT: Back to original direction per user request
        steering = np.clip(action[1], -1.0, 1.0) * 0.75   # Back to Positive
        
        # Apply to robot
        self._apply_action(throttle, steering)
        
        # Step simulation
        self.world.step(render=True)
        
        # Get robot position for geofence and goal check
        import isaacsim.core.utils.xforms as xforms_utils
        # Use Chassis for actual physics position
        robot_pos, _ = xforms_utils.get_world_pose(prim_path="/World/Leatherback/Rigid_Bodies/Chassis")
        
        # Initialize termination and truncation flags
        terminated = False
        truncated = False
        
        # Get observation to extract current distance to waypoint
        obs = self._get_observation()
        dist_to_goal = obs["vector"][0] # This is the distance to the current waypoint
        
        # Reward calculation (simplified for this example)
        reward = -dist_to_goal 
        
        # Reached goal?
        if dist_to_goal < self.position_tolerance: # Use existing tolerance
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx < self.num_waypoints:
                # Move marker to next waypoint
                next_wp = self.waypoints[self.current_waypoint_idx]
                self.goal_marker.set_world_pose(position=np.array([next_wp[0], next_wp[1], 0.3]))
                print(f"✓ Waypoint reached! Next: {next_wp}")
            
            reward += 50.0 # Goal bonus
            print(f"✓ Waypoint {self.current_waypoint_idx}/{self.num_waypoints} reached!")
            
            if self.current_waypoint_idx >= self.num_waypoints:
                print("All waypoints reached!")
                terminated = True
            
        # Virtual Geofence (Replace Walls)
        # If robot goes > 12m from origin (arena was 20x20, so 10m radius usually)
        dist_from_origin = np.linalg.norm(robot_pos[:2])
        if dist_from_origin > 12.0:
            reward -= 50.0 # Heavy penalty for leaving arena
            terminated = True
            print("Left Arena (Geofence Violation) - Resetting")
            
        # Flip Check (Safety)
        # Check standard Z-up vector from orientation
        # orientation is quaternion [w, x, y, z]
        _, rot = xforms_utils.get_world_pose(prim_path="/World/Leatherback/Rigid_Bodies/Chassis")
        # Rotate up vector (0,0,1) by quaternion
        # Simplified: Check if Z component of local up is negative (upside down)
        # Actually just check roll/pitch from quat, or raw Z height if it drops too low (falling)
        if robot_pos[2] < 0.0: # Falling through floor
             terminated = True
             reward -= 50.0
             print("Fell through world!")
             
        # Check if flipped (Up axis pointing down)
        # q = [w, x, y, z]
        # z-axis transform roughly: 1 - 2(x^2 + y^2) 
        # If this is negative, we are > 90 deg tipped
        w, x, y, z = rot
        z_axis_z = 1.0 - 2.0 * (x*x + y*y)
        if z_axis_z < 0.3: # Tilted more than ~70 degrees
             terminated = True
             reward -= 50.0
             print("Robot Flipped! Resetting.")
        self.prev_distance = dist_to_goal
            
        return obs, reward, terminated, truncated, {}
    
    def close(self):
        """Cleanup"""
        if hasattr(self, 'sim_app'):
            self.sim_app.close()


if __name__ == "__main__":
    print("="*60)
    print("Testing Leatherback Environment")
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    parser.add_argument("--camera", action="store_true", help="Enable RGB camera")
    args = parser.parse_args()
    
    # Initialize environment
    # Default to NO camera to prevent crash for now
    env = LeatherbackEnv(headless=args.headless, use_camera=args.camera)
    
    print("\nResetting environment...")
    obs, info = env.reset()
    # print(f"Observation shape: {obs.shape}") # No longer valid for Dict
    print(f"Initial observation keys: {obs.keys()}")
    
    # Test loop
    print("\nRunning simulation (Press Ctrl+C to stop)...")
    try:
        for i in range(1000):
            if not env.sim_app.is_running():
                break
                
            # Random action: [throttle, steering]
            # Throttle: 1.0 (Forward)
            # Steering: sine wave to wiggle
            action = np.array([1.0, np.sin(i * 0.1)], dtype=np.float32)
            
            obs, reward, done, truncated, info = env.step(action)
            
            if i % 10 == 0:
                dist = obs["vector"][0]
                print(f"Step {i}: Dist={dist:.2f}m, Reward={reward:.3f}")
                
            if done:
                print("Goal reached or reset triggered!")
                env.reset()
                
    except KeyboardInterrupt:
        print("User stopped simulation.")
        
    print("\nClosing environment...")
    env.close()
    print("✓ Test complete!")

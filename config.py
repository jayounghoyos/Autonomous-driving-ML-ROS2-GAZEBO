"""
Central Configuration for Autonomous Driving Project
Stores all hyperparameters, reward weights, and robot settings.
"""

class RobotConfig:
    # Camera Settings
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84
    IMAGE_CHANNELS = 3
    
    # Drive Settings (Ackermann - Scaled 60%)
    MAX_VELOCITY = 2.0
    MIN_VELOCITY = -2.0
    MAX_STEER = 0.7
    MIN_STEER = -0.7
    
    # Topics
    TOPIC_IMAGE = '/camera/image_raw'
    TOPIC_ODOM = '/odom'
    TOPIC_CMD_VEL = '/cmd_vel'
    TOPIC_SCAN = '/scan'
    
    # Legacy (4-Wheel) - Deprecated
    # TOPIC_FRONT_LEFT = ...
    
    # Simulation Control
    TOPIC_GOAL_POSE = '/model/goal_marker/pose'

class RewardConfig:
    # Goal Rewards
    GOAL_REACHED_BONUS = 200.0
    DISTANCE_IMPROVEMENT_MULTIPLIER = 20.0
    
    # Vision Rewards
    VISION_VISIBLE_BONUS = 10.0   # Reward for seeing the goal
    VISION_CENTER_BONUS = 5.0     # Reward for centering the goal
    
    # Penalties
    TIME_PENALTY = 0.05
    SPIN_PENALTY_WEIGHT = 0.5
    REVERSE_PENALTY_WEIGHT = 0.1  # Greatly reduced (was 2.0)
    HEADING_PENALTY_WEIGHT = 10.0 # Penalty for looking away
    BLIND_PENALTY = 0.05          # Small penalty for not seeing goal
    STUCK_PENALTY_WEIGHT = 0.5    # Penalty for pushing but not moving
    COLLISION_PENALTY = 50.0      # Instant Game Over

    # Thresholds
    STUCK_VELOCITY_THRESHOLD = 0.01
    GOAL_DISTANCE_THRESHOLD = 0.5 # Meters (GPS)
    GOAL_VISION_THRESHOLD = 0.4   # Screen area ratio (Vision)
    COLLISION_DISTANCE = 0.25     # Meters (LiDAR min)
    LIDAR_RAYS = 16               # Number of rays for observation
    
    # Boundary Settings (Raceway)
    # BOUNDARY_X = (-300.0, 300.0) 
    # BOUNDARY_Y = (-150.0, 150.0)
    
    # Boundary Settings (Raceway)
    # BOUNDARY_X = (-300.0, 300.0) 
    # BOUNDARY_Y = (-150.0, 150.0)
    
    # Boundary Settings (Gymkhana/Warehouse)
    BOUNDARY_X = (-50.0, 50.0)
    BOUNDARY_Y = (-50.0, 50.0)
    OUT_OF_BOUNDS_PENALTY = -100.0

    # Goal Settings
    GOAL_REACH_DISTANCE = 1.0 
    COLLISION_PENALTY = -50.0
    TIME_PENALTY = -0.05
    GOAL_REWARD = 100.0
    
    # Random Goal Spawn Range
    GOAL_X_RANGE = (-5.0, 40.0) # Covering most of the gymkhana track
    GOAL_Y_RANGE = (-20.0, 20.0)
    
    # Waypoint System (Raceway Mode - 38 Degree Right Turn)
    # Path: Diagonally Right (Angle -38 deg)
    # Gymkhana Waypoints (Progressive Obstacle Course)
    WAYPOINTS = [
        (0.0, 0.0),        # Start
        (10.0, 0.0),       # WP1: After slalom (Easy)
        (28.0, 0.0),       # WP2: Through narrow passage (Medium)
        (40.0, 0.0),       # WP3: Through box maze (Hard)
        (52.0, 0.0),       # Goal: Final sprint
    ]
    
    WAYPOINT_RADIUS = 3.0  # Reduced radius for tighter room tracking
    WAYPOINT_REWARD = 50.0
    LAP_COMPLETION_BONUS = 500.0

class TrainingConfig:
    # PPO Hyperparameters
    LEARNING_RATE = 1e-4
    N_STEPS = 2048
    BATCH_SIZE = 32
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01
    
    # Training Setup
    TOTAL_TIMESTEPS = 500_000
    CHECKPOINT_FREQ = 10_000
    DEVICE = 'cuda'
    VERBOSE = 1

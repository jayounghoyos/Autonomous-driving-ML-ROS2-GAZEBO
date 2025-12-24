# Autonomous Driving with Isaac Sim 5.1.0 + Isaac Lab

Reinforcement learning for autonomous vehicle navigation using NVIDIA Isaac Sim and Isaac Lab with the Leatherback vehicle model.

## Requirements

| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 LTS |
| Python | 3.11 |
| NVIDIA Driver | 580.65.06+ |
| CUDA | 12.8 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | Latest |
| ROS2 | Jazzy |

**Hardware:**
- NVIDIA GPU with 16GB+ VRAM
- 32GB+ RAM
- 50GB+ disk space

## Quick Start

### 1. Isaac Sim 5.1.0 (Standalone - Already Installed)

Your Isaac Sim is installed at: `~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64`

### 2. Install Isaac Lab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
./isaaclab.sh --install sb3
```

### 3. Source Environment

```bash
source scripts/setup_isaac_env.sh
```

### 4. Train

```bash
# Headless training (faster)
$ISAAC_PYTHON training/train_ppo.py --headless

# With GUI (watch the robot learn)
$ISAAC_PYTHON training/train_ppo.py
```

### 5. Evaluate

```bash
$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip
```

## Project Structure

```
.
├── isaac_lab/              # Isaac Lab environment integration
│   └── envs/
│       ├── leatherback_env.py      # Main environment
│       └── leatherback_env_cfg.py  # Configuration
├── training/               # Training pipeline
│   ├── train_ppo.py        # PPO training script
│   ├── evaluate.py         # Evaluation script
│   └── configs/            # YAML configurations
├── ros2_ws/                # ROS2 Jazzy workspace
│   └── src/
│       ├── isaac_ros_bridge/   # Isaac <-> ROS2 bridge
│       ├── autonomous_nav/     # Navigation algorithms
│       ├── ml_perception/      # Perception (YOLO, etc.)
│       └── vehicle_control/    # Vehicle control
├── scripts/                # Setup scripts
│   └── setup_isaac_env.sh  # Environment setup
├── models/                 # Trained models
└── logs/                   # Training logs
```

## Robot Model

Uses NVIDIA's **Leatherback** vehicle - a 4-wheeled Ackermann steering robot ideal for autonomous driving research.

## Training Configuration

Edit `training/configs/ppo_config.yaml` to customize:

- `num_envs`: Parallel environments (1 for SB3)
- `total_timesteps`: Training duration
- `learning_rate`: PPO learning rate
- `goal_tolerance`: Distance to waypoint
- `num_waypoints`: Waypoints per episode

## ROS2 Integration

Build and source the ROS2 workspace:

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

## License

MIT License

## References

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/main/index.html)
- [Isaac Sim 5.1.0 ROS2](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_ros.html)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)

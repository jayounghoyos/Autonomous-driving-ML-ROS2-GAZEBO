# Quick Start Guide

Complete setup guide for the Leatherback autonomous driving project using Isaac Sim 5.1.0, Isaac Lab, and ROS2 Jazzy.

## Prerequisites

- Ubuntu 24.04 LTS
- NVIDIA GPU with 16GB+ VRAM
- Python 3.11
- NVIDIA Driver 580.65.06+

## Step 1: System Preparation

### Install NVIDIA Driver

```bash
# Check current driver
nvidia-smi

# If needed, install latest driver
sudo apt install nvidia-driver-580
sudo reboot
```

### Install Python 3.11

```bash
sudo apt install python3.11 python3.11-venv python3.11-dev
```

### Install ROS2 Jazzy

```bash
# Add ROS2 repository
sudo apt update && sudo apt install software-properties-common
sudo add-apt-repository universe
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
sudo sh -c 'echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'

# Install ROS2 Jazzy
sudo apt update
sudo apt install ros-jazzy-desktop ros-jazzy-vision-msgs ros-jazzy-ackermann-msgs

# Add to bashrc
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## Step 2: Isaac Sim 5.1.0 (Standalone)

Your Isaac Sim is already installed at:
```
~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64
```

### Verify Installation

```bash
# Launch Isaac Sim
~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/isaac-sim.sh
```

## Step 3: Isaac Lab Setup

### Clone and Install

```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab

# Install with Stable-Baselines3 support
./isaaclab.sh --install sb3
```

### Verify Isaac Lab

```bash
./isaaclab.sh -p scripts/tutorials/00_sim/create_empty.py
```

## Step 4: Project Setup

### Source Environment

```bash
cd ~/Autonomous-driving-ML-ROS2-GAZEBO
source scripts/setup_isaac_env.sh
```

This sets up:
- `$ISAAC_SIM_PATH` - Path to Isaac Sim
- `$ISAAC_PYTHON` - Isaac Sim's Python interpreter
- `$ISAACLAB_PATH` - Path to Isaac Lab
- ROS2 Jazzy environment

## Step 5: Training

### Quick Training Test

```bash
# Test environment (100 steps)
$ISAAC_PYTHON isaac_lab/envs/leatherback_env.py --headless --steps 100
```

### Full Training (Headless)

```bash
$ISAAC_PYTHON training/train_ppo.py --headless --timesteps 500000
```

### Training with GUI

```bash
$ISAAC_PYTHON training/train_ppo.py
```

### Monitor Training

```bash
# In another terminal
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006 in browser
```

## Step 6: Evaluation

```bash
$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip --episodes 10
```

## Step 7: ROS2 Integration (Optional)

### Build ROS2 Workspace

```bash
cd ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

### Launch Isaac-ROS Bridge

```bash
# Terminal 1: Isaac Sim with ROS2
ros2 run isaac_ros_bridge isaac_publisher

# Terminal 2: Send commands
ros2 topic pub /cmd_vel geometry_msgs/Twist "{linear: {x: 1.0}, angular: {z: 0.5}}"
```

## What to Expect

### Training Phases

| Phase | Timesteps | Behavior | Reward |
|-------|-----------|----------|--------|
| 1: Random | 0-10k | Random exploration, collisions | -50 to 0 |
| 2: Learning | 10k-50k | Avoids walls, sometimes reaches goal | 0 to +20 |
| 3: Improving | 50k-200k | Regular goal reaching | +20 to +50 |
| 4: Converging | 200k-500k | Optimal paths, 80%+ success | +50 to +80 |

## Troubleshooting

### Isaac Sim Won't Start

```bash
# Check GLIBC version (needs 2.35+)
ldd --version

# Check GPU memory
nvidia-smi
```

### Python Version Mismatch

```bash
# Verify Python 3.11
python --version

# If wrong, activate correct environment
source ~/env_isaaclab/bin/activate
```

### ROS2 Not Found

```bash
# Source ROS2
source /opt/ros/jazzy/setup.bash

# Verify
ros2 --version
```

### Training Crashes

- Reduce `num_envs` in config
- Enable `--headless` mode
- Check GPU memory with `nvidia-smi`

## Next Steps

1. Modify reward function in `isaac_lab/envs/leatherback_env_cfg.py`
2. Add obstacles or complexity to training
3. Try different RL algorithms (SAC, TD3)
4. Deploy trained model to real robot via ROS2

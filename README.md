# Autonomous Driving ML with ROS2 & Gazebo

Goal-seeking robot trained with PPO reinforcement learning.

## Quick Start

### 1. Simple Training (Watch the robot learn!)
```bash
# Terminal 1: Launch Gazebo GUI
source ~/ros2-ml-env/bin/activate
source ~/PersonalPorjects/Autonomous-driving-ML-ROS2-GAZEBO/install/setup.zsh
./launch_with_gui.sh

# Terminal 2: Train
source ~/ros2-ml-env/bin/activate
python3 train_simple.py
```

### 2. Parallel Training (Maximum Speed!)
```bash
source ~/ros2-ml-env/bin/activate

# 12 parallel environments (recommended)
python3 train_parallel.py --num-envs 12 --timesteps 1000000
```

## Files

**Training:**
- `train_simple.py` - Single environment training (watch in GUI)
- `train_parallel.py` - Multi-environment parallel training
- `goal_seeking_env.py` - Gym environment implementation

**Simulation:**
- `simple_robot.sdf` - Robot model (2-wheel differential drive)
- `launch_with_gui.sh` - Launch Gazebo with GUI
- `launch_headless.sh` - Launch Gazebo headless
- `drive_simple_robot.py` - Manual WASD control

**Utilities:**
- `kill_gazebo.sh` - Kill all Gazebo processes
- `TRAINING_COMMANDS.md` - Full training reference

## System Requirements
- ROS 2 Kilted
- Gazebo Sim
- CUDA GPU (optional but recommended)
- 16GB+ RAM for parallel training

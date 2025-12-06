# Training Commands

## Simple Training (1 environment, watch in GUI)
```bash
# Terminal 1: Launch Gazebo GUI
source ~/ros2-ml-env/bin/activate
source ~/PersonalPorjects/Autonomous-driving-ML-ROS2-GAZEBO/install/setup.zsh
./launch_with_gui.sh

# Terminal 2: Train
source ~/ros2-ml-env/bin/activate
python3 train_simple.py
```

## Parallel Training (Maximum Speed!)
```bash
# Single terminal - all automated
source ~/ros2-ml-env/bin/activate

# 8 parallel environments (recommended for 16GB RAM)
python3 train_parallel.py --num-envs 8 --timesteps 500000

# 16 parallel environments (if you have 32GB+ RAM)
python3 train_parallel.py --num-envs 16 --timesteps 1000000

# Custom
python3 train_parallel.py --num-envs 12 --timesteps 750000
```

## System Requirements
- **8 envs:** 16GB RAM, 8 CPU cores
- **16 envs:** 32GB RAM, 16 CPU cores
- **GPU:** RTX 5060 Ti (you have this!)

## Expected Training Times
- **Simple (1 env):** 500k steps = ~2 hours
- **Parallel (8 envs):** 500k steps = ~15 minutes 
- **Parallel (16 envs):** 1M steps = ~15 minutes 

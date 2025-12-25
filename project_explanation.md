# Project Explanation: Autonomous Driving with Isaac Sim & Isaac Lab

> **Last Updated**: 2025-12-24  
> **Project**: Leatherback Autonomous Driving RL Training  
> **Tech Stack**: Isaac Sim 5.1.0, Isaac Lab, ROS2 Jazzy, Python 3.11

---

## Table of Contents

### Batch 1: Root Configuration Files
- [README.md](#readmemd)
- [QUICK_START.md](#quick_startmd)
- [pyproject.toml](#pyprojecttoml)
- [requirements.txt](#requirementstxt)
- [.gitignore](#gitignore)

### Batch 2: Setup Scripts
- [scripts/setup_isaac_env.sh](#scriptssetup_isaac_envsh)
- [scripts/verify_installation.py](#scriptsverify_installationpy)

### Batch 3: Isaac Lab Environment Core
- [isaac_lab/__init__.py](#isaac_lab__init__py)
- [isaac_lab/envs/__init__.py](#isaac_labenvs__init__py)
- [isaac_lab/envs/leatherback_env.py](#isaac_labenvsleatherback_envpy)
- [isaac_lab/envs/leatherback_env_cfg.py](#isaac_labenvsleatherback_env_cfgpy)
- [isaac_lab/tasks/__init__.py](#isaac_labtasks__init__py)

### Batch 4: Training Pipeline
- [training/__init__.py](#training__init__py)
- [training/train_ppo.py](#trainingtrain_ppopy)
- [training/evaluate.py](#trainingevaluatepy)
- [training/view_agent.py](#trainingview_agentpy)
- [training/configs/ppo_config.yaml](#trainingconfigsppo_configyaml)
- [training/configs/env_config.yaml](#trainingconfigsenv_configyaml)

### Batch 5: ROS2 Bridge Package
- [ros2_ws/src/isaac_ros_bridge/package.xml](#ros2_wssrcisaac_ros_bridgepackagexml)
- [ros2_ws/src/isaac_ros_bridge/setup.py](#ros2_wssrcisaac_ros_bridgesetuppy)
- [ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/__init__.py](#ros2_wssrcisaac_ros_bridgeisaac_ros_bridge__init__py)
- [ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/isaac_publisher.py](#ros2_wssrcisaac_ros_bridgeisaac_ros_bridgeisaac_publisherpy)
- [ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/ros_subscriber.py](#ros2_wssrcisaac_ros_bridgeisaac_ros_bridgeros_subscriberpy)

### Batch 6: Vehicle Control Package
- [ros2_ws/src/vehicle_control/package.xml](#ros2_wssrcvehicle_controlpackagexml)
- [ros2_ws/src/vehicle_control/setup.py](#ros2_wssrcvehicle_controlsetuppy)
- [ros2_ws/src/vehicle_control/vehicle_control/__init__.py](#ros2_wssrcvehicle_controlvehicle_control__init__py)
- [ros2_ws/src/vehicle_control/vehicle_control/pid_controller.py](#ros2_wssrcvehicle_controlvehicle_controlpid_controllerpy)
- [ros2_ws/src/vehicle_control/vehicle_control/teleop_keyboard.py](#ros2_wssrcvehicle_controlvehicle_controlteleop_keyboardpy)
- [ros2_ws/src/vehicle_control/vehicle_control/teleop_keyboard_ackermann.py](#ros2_wssrcvehicle_controlvehicle_controlteleop_keyboard_ackermannpy)

### Batch 7: ML Perception Package
- [ros2_ws/src/ml_perception/package.xml](#ros2_wssrcml_perceptionpackagexml)
- [ros2_ws/src/ml_perception/setup.py](#ros2_wssrcml_perceptionsetuppy)
- [ros2_ws/src/ml_perception/ml_perception/__init__.py](#ros2_wssrcml_perceptionml_perception__init__py)
- [ros2_ws/src/ml_perception/ml_perception/yolo_detector.py](#ros2_wssrcml_perceptionml_perceptionyolo_detectorpy)
- [ros2_ws/src/ml_perception/ml_perception/lane_detector.py](#ros2_wssrcml_perceptionml_perceptionlane_detectorpy)
- [ros2_ws/src/ml_perception/ml_perception/waymo_parser.py](#ros2_wssrcml_perceptionml_perceptionwaymo_parserpy)
- [ros2_ws/src/ml_perception/ml_perception/dataset_preprocessor.py](#ros2_wssrcml_perceptionml_perceptiondataset_preprocessorpy)

### Batch 8: Autonomous Navigation Package
- [ros2_ws/src/autonomous_nav/package.xml](#ros2_wssrcautonomous_navpackagexml)
- [ros2_ws/src/autonomous_nav/setup.py](#ros2_wssrcautonomous_navsetuppy)
- [ros2_ws/src/autonomous_nav/autonomous_nav/__init__.py](#ros2_wssrcautonomous_navautonomous_nav__init__py)
- [ros2_ws/src/autonomous_nav/autonomous_nav/lane_follower.py](#ros2_wssrcautonomous_navautonomous_navlane_followerpy)

### Batch 9: RL Training Package
- [ros2_ws/src/rl_training/package.xml](#ros2_wssrcrl_trainingpackagexml)
- [ros2_ws/src/rl_training/setup.py](#ros2_wssrcrl_trainingsetuppy)
- [ros2_ws/src/rl_training/rl_training/__init__.py](#ros2_wssrcrl_trainingrl_training__init__py)
- [ros2_ws/src/rl_training/rl_training/rl_agent.py](#ros2_wssrcrl_trainingrl_trainingrl_agentpy)

---

## Documentation Status

**Batches Completed**: 9/9 ✅ 🎉  
**Files Documented**: 44+  
**Completion Date**: 2025-12-24

---

## How to Use This Document

This document provides a comprehensive explanation of every important file in the project. Each file section includes:

1. **Purpose**: What the file does
2. **Key Components**: Important classes, functions, or configurations
3. **Dependencies**: What it relies on
4. **Usage**: How it's used in the project
5. **Important Code Sections**: Detailed explanations of critical parts

---

## Batch Progress Tracker

- [x] **Batch 1**: Root Configuration Files (5 files) ✅
- [x] **Batch 2**: Setup Scripts (2 files) ✅
- [x] **Batch 3**: Isaac Lab Environment Core (5 files) ✅ 🔥
- [x] **Batch 4**: Training Pipeline (6 files) ✅
- [x] **Batch 5**: ROS2 Bridge Package (5 files) ✅
- [x] **Batch 6**: Vehicle Control Package (6 files) ✅
- [x] **Batch 7**: ML Perception Package (7 files) ✅
- [x] **Batch 8**: Autonomous Navigation Package (4 files) ✅
- [x] **Batch 9**: RL Training Package (4 files) ✅

**🎉 ALL BATCHES COMPLETE! 🎉**







---

# File Documentation

<!-- Documentation will be added batch by batch below -->

---

# BATCH 1: ROOT CONFIGURATION FILES

**Status**: ✅ COMPLETED  
**Files**: 5  
**Date**: 2025-12-24

---

## README.md

**Path**: `/README.md`  
**Type**: Documentation  
**Lines**: 117  
**Purpose**: Main project documentation providing overview, requirements, quick start, and project structure

### Key Sections

#### 1. Project Title and Description (Lines 1-3)
```markdown
# Autonomous Driving with Isaac Sim 5.1.0 + Isaac Lab
Reinforcement learning for autonomous vehicle navigation using NVIDIA Isaac Sim and Isaac Lab with the Leatherback vehicle model.
```
**Explanation**: Clearly states the project uses the latest Isaac Sim 5.1.0 with Isaac Lab framework, focusing on the Leatherback vehicle for autonomous driving RL training.

#### 2. Requirements Table (Lines 5-15)
```markdown
| Component | Version |
|-----------|---------|
| OS | Ubuntu 24.04 LTS |
| Python | 3.11 |
| NVIDIA Driver | 580.65.06+ |
| CUDA | 12.8 |
| Isaac Sim | 5.1.0 |
| Isaac Lab | Latest |
| ROS2 | Jazzy |
```
**Explanation**: 
- **Python 3.11** is mandatory for Isaac Sim 5.x (5.0+ requires 3.11, older versions used 3.10)
- **NVIDIA Driver 580.65.06+** is the production branch recommended for Isaac Sim 5.1.0
- **CUDA 12.8** matches the PyTorch build requirements
- **ROS2 Jazzy** is natively supported on Ubuntu 24.04

#### 3. Hardware Requirements (Lines 17-20)
```markdown
- NVIDIA GPU with 16GB+ VRAM
- 32GB+ RAM
- 50GB+ disk space
```
**Explanation**: These are Isaac Lab's minimum requirements. The 16GB VRAM is needed for parallel environment rendering (4096 envs), and 32GB RAM handles the simulation physics.

#### 4. Quick Start Commands (Lines 42-50)
```bash
# Headless training (faster)
$ISAAC_PYTHON training/train_ppo.py --headless

# With GUI (watch the robot learn)
$ISAAC_PYTHON training/train_ppo.py
```
**Explanation**: Uses `$ISAAC_PYTHON` environment variable (set by `setup_isaac_env.sh`) which points to Isaac Sim's bundled Python 3.11 interpreter with all necessary libraries.

#### 5. Project Structure (Lines 58-80)
**Explanation**: Shows the reorganized structure:
- `isaac_lab/` - Core environment implementation
- `training/` - Training scripts and configs
- `ros2_ws/` - ROS2 Jazzy workspace (separate from Isaac)
- `scripts/` - Setup utilities
- `models/` and `logs/` - Artifacts

#### 6. Robot Model Section (Lines 82-84)
```markdown
Uses NVIDIA's **Leatherback** vehicle - a 4-wheeled Ackermann steering robot ideal for autonomous driving research.
```
**Explanation**: Leatherback is NVIDIA's official Ackermann vehicle model, available in Isaac Sim's nucleus assets. It has realistic steering geometry and suspension.

#### 7. Training Configuration (Lines 86-94)
**Explanation**: Points users to `ppo_config.yaml` for hyperparameter tuning. Key parameters:
- `num_envs`: Set to 1 for Stable-Baselines3 (SB3 doesn't support vectorized Isaac envs natively)
- `goal_tolerance`: Distance threshold for waypoint reaching
- `num_waypoints`: Number of navigation goals per episode

#### 8. References (Lines 112-116)
**Explanation**: Links to official documentation:
- Isaac Lab main docs
- Isaac Sim 5.1.0 ROS2 integration guide
- Stable-Baselines3 (the RL library used)

### Dependencies
- None (pure documentation)

### Usage
First file users read to understand the project.

---

## QUICK_START.md

**Path**: `/QUICK_START.md`  
**Type**: Tutorial Documentation  
**Lines**: 207  
**Purpose**: Step-by-step installation and setup guide for new users

### Key Sections

#### 1. Prerequisites (Lines 5-10)
**Explanation**: Lists system requirements before starting. Ubuntu 24.04 is required for native ROS2 Jazzy support.

#### 2. System Preparation (Lines 12-47)

##### NVIDIA Driver Installation (Lines 14-23)
```bash
nvidia-smi  # Check current driver
sudo apt install nvidia-driver-580
sudo reboot
```
**Explanation**: Driver 580 is the production branch. The reboot is necessary for the driver to load properly.

##### Python 3.11 Installation (Lines 25-29)
```bash
sudo apt install python3.11 python3.11-venv python3.11-dev
```
**Explanation**: 
- `python3.11-venv` - For creating virtual environments
- `python3.11-dev` - Headers needed for compiling Python extensions (required by some Isaac Lab dependencies)

##### ROS2 Jazzy Installation (Lines 31-47)
```bash
sudo apt install ros-jazzy-desktop ros-jazzy-vision-msgs ros-jazzy-ackermann-msgs
```
**Explanation**:
- `ros-jazzy-desktop` - Full ROS2 desktop install
- `ros-jazzy-vision-msgs` - For camera/sensor messages
- `ros-jazzy-ackermann-msgs` - For Ackermann steering control messages

#### 3. Isaac Sim 5.1.0 Standalone (Lines 49-61)
```bash
~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/isaac-sim.sh
```
**Explanation**: The standalone version is already installed. This is the binary distribution (not pip). The pip version would be installed via `pip install isaacsim[all,extscache]==5.1.0`.

#### 4. Isaac Lab Setup (Lines 63-79)
```bash
git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab
cd ~/IsaacLab
./isaaclab.sh --install sb3
```
**Explanation**:
- Clones Isaac Lab to home directory
- `--install sb3` installs Stable-Baselines3 integration
- Other options: `rl_games`, `rsl_rl`, `skrl`, `robomimic`

#### 5. Environment Setup (Lines 82-94)
```bash
source scripts/setup_isaac_env.sh
```
**Explanation**: This script sets critical environment variables:
- `$ISAAC_SIM_PATH` - Path to Isaac Sim installation
- `$ISAAC_PYTHON` - Isaac Sim's Python interpreter
- `$ISAACLAB_PATH` - Path to Isaac Lab
- Sources ROS2 Jazzy

#### 6. Training Commands (Lines 96-123)

##### Quick Test (Lines 100-103)
```bash
$ISAAC_PYTHON isaac_lab/envs/leatherback_env.py --headless --steps 100
```
**Explanation**: Runs the environment standalone for 100 steps to verify it works. This doesn't train, just tests the environment.

##### Full Training (Lines 105-115)
```bash
$ISAAC_PYTHON training/train_ppo.py --headless --timesteps 500000
```
**Explanation**:
- `--headless` disables GUI rendering (2-3x faster)
- `--timesteps 500000` trains for 500k steps (about 2-4 hours depending on GPU)

##### TensorBoard Monitoring (Lines 117-123)
```bash
tensorboard --logdir logs/tensorboard
```
**Explanation**: Opens TensorBoard on port 6006 to visualize training metrics in real-time.

#### 7. Training Phases Table (Lines 152-161)
```markdown
| Phase | Timesteps | Behavior | Reward |
|-------|-----------|----------|--------|
| 1: Random | 0-10k | Random exploration, collisions | -50 to 0 |
| 2: Learning | 10k-50k | Avoids walls, sometimes reaches goal | 0 to +20 |
| 3: Improving | 50k-200k | Regular goal reaching | +20 to +50 |
| 4: Converging | 200k-500k | Optimal paths, 80%+ success | +50 to +80 |
```
**Explanation**: Sets expectations for training progress. Users can compare their training curves to these benchmarks.

#### 8. Troubleshooting (Lines 163-199)

##### GLIBC Version Check (Lines 168-170)
```bash
ldd --version  # Needs 2.35+
```
**Explanation**: Isaac Sim 5.x requires GLIBC 2.35+, which is why Ubuntu 24.04 is required (Ubuntu 22.04 has 2.34).

##### Python Version Verification (Lines 176-183)
**Explanation**: Common issue is using system Python instead of Isaac Sim's Python. The `$ISAAC_PYTHON` variable should always be used.

### Dependencies
- None (pure documentation)

### Usage
Follow this guide sequentially for first-time setup.

---

## pyproject.toml

**Path**: `/pyproject.toml`  
**Type**: Python Project Configuration (PEP 518/621)  
**Lines**: 77  
**Purpose**: Modern Python packaging configuration defining project metadata, dependencies, and development tools

### Key Sections

#### 1. Project Metadata (Lines 1-18)
```toml
[project]
name = "leatherback-rl"
version = "2.0.0"
description = "Autonomous driving RL with Isaac Sim 5.1.0 and Isaac Lab"
requires-python = ">=3.11"
```
**Explanation**:
- **Version 2.0.0** indicates major refactor from Gazebo to Isaac Sim
- **requires-python = ">=3.11"** enforces Python 3.11+ (Isaac Sim 5.x requirement)
- **MIT License** allows free use and modification

#### 2. Keywords and Classifiers (Lines 11-18)
```toml
keywords = ["autonomous-driving", "reinforcement-learning", "isaac-sim", "isaac-lab", "ros2"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Programming Language :: Python :: 3.11",
]
```
**Explanation**: Helps with discoverability if published to PyPI. "Alpha" status indicates active development.

#### 3. Core Dependencies (Lines 20-28)
```toml
dependencies = [
    "torch>=2.7.0",
    "gymnasium>=1.0.0",
    "stable-baselines3>=2.4.0",
    "numpy>=1.26.0",
    "pyyaml>=6.0.2",
    "tensorboard>=2.18.0",
    "tqdm>=4.67.0",
]
```
**Explanation**:
- **torch>=2.7.0** - PyTorch 2.7 with CUDA 12.8 support
- **gymnasium>=1.0.0** - Modern Gym API (replaces old `gym` package)
- **stable-baselines3>=2.4.0** - PPO, SAC, TD3 implementations
- **pyyaml** - For loading config files
- **tensorboard** - Training visualization
- **tqdm** - Progress bars

**Note**: Isaac Sim is NOT listed here because it's installed separately via pip with special index URL.

#### 4. Optional Dependencies (Lines 30-40)
```toml
[project.optional-dependencies]
dev = ["pytest>=8.3.0", "black>=24.10.0", "isort>=5.13.0", "mypy>=1.13.0"]
perception = ["opencv-python>=4.10.0", "ultralytics>=8.3.0"]
```
**Explanation**:
- **dev**: Development tools (testing, formatting, type checking)
  - Install with: `pip install -e ".[dev]"`
- **perception**: Computer vision tools (YOLO, OpenCV)
  - Install with: `pip install -e ".[perception]"`

#### 5. Build System (Lines 45-47)
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"
```
**Explanation**: Uses modern setuptools for building the package. Allows `pip install -e .` for editable installs.

#### 6. Package Discovery (Lines 49-51)
```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["isaac_lab*", "training*"]
```
**Explanation**: Tells setuptools to include `isaac_lab/` and `training/` directories when packaging. ROS2 packages in `ros2_ws/` are excluded (they have their own setup).

#### 7. Black Configuration (Lines 53-65)
```toml
[tool.black]
line-length = 100
target-version = ["py311"]
exclude = '''/(\.git | \.venv | build | dist | logs | models)/'''
```
**Explanation**:
- **line-length = 100** - Maximum line length (default is 88)
- **target-version = ["py311"]** - Optimize for Python 3.11 syntax
- **exclude** - Don't format generated files, logs, or models

#### 8. isort Configuration (Lines 67-70)
```toml
[tool.isort]
profile = "black"
line_length = 100
```
**Explanation**: Import sorting compatible with Black formatter. Keeps imports organized.

#### 9. mypy Configuration (Lines 72-77)
```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
ignore_missing_imports = true
```
**Explanation**:
- **ignore_missing_imports = true** - Necessary because Isaac Sim packages don't have type stubs
- **warn_return_any** - Helps catch type errors

### Dependencies
- None (configuration file)

### Usage
```bash
# Install project in editable mode
pip install -e .

# Install with dev tools
pip install -e ".[dev]"

# Format code
black isaac_lab/ training/

# Sort imports
isort isaac_lab/ training/

# Type check
mypy isaac_lab/
```

---

## requirements.txt

**Path**: `/requirements.txt`  
**Type**: Python Dependencies List  
**Lines**: 37  
**Purpose**: Pip-installable dependencies for the project (alternative to pyproject.toml)

### Key Sections

#### 1. Header Comments (Lines 1-2)
```txt
# Leatherback RL - Isaac Sim 5.1.0 + Isaac Lab
# Python 3.11 required
```
**Explanation**: Reminds users of the Python version requirement.

#### 2. Core ML Dependencies (Lines 4-7)
```txt
torch==2.7.0
torchvision>=0.22.0
torchaudio>=2.7.0
```
**Explanation**:
- **torch==2.7.0** - Exact version for CUDA 12.8 compatibility
- **torchvision/torchaudio** - Use `>=` to allow minor updates
- Install command: `pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128`

#### 3. RL Dependencies (Lines 9-12)
```txt
gymnasium>=1.0.0
stable-baselines3>=2.4.0
sb3-contrib>=2.4.0
```
**Explanation**:
- **gymnasium** - OpenAI Gym successor (modern API)
- **stable-baselines3** - Main RL library (PPO, SAC, TD3)
- **sb3-contrib** - Additional algorithms (TQC, QRDQN)

#### 4. Isaac Sim Installation Note (Lines 14-15)
```txt
# Isaac Sim 5.1.0 - Install separately via:
# pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```
**Explanation**: Isaac Sim is NOT in requirements.txt because:
1. It requires a special NVIDIA PyPI index
2. It's a large download (~10GB)
3. Users might already have it installed standalone

**Installation command**:
```bash
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

#### 5. Utilities (Lines 17-24)
```txt
numpy>=1.26.0
scipy>=1.14.0
matplotlib>=3.9.0
tensorboard>=2.18.0
pyyaml>=6.0.2
tqdm>=4.67.0
pandas>=2.2.0
```
**Explanation**:
- **numpy/scipy** - Numerical computing
- **matplotlib** - Plotting (for evaluation visualizations)
- **tensorboard** - Training metrics visualization
- **pyyaml** - YAML config file parsing
- **tqdm** - Progress bars
- **pandas** - Data manipulation (for logging)

#### 6. Perception (Lines 26-28)
```txt
opencv-python>=4.10.0
ultralytics>=8.3.0
```
**Explanation**:
- **opencv-python** - Computer vision operations
- **ultralytics** - YOLOv8/v11 for object detection
- These are optional (only needed for perception package)

#### 7. ROS2 Note (Lines 30-31)
```txt
# ROS2 Jazzy - Install system packages via apt, not pip
# sudo apt install ros-jazzy-desktop ros-jazzy-vision-msgs ros-jazzy-ackermann-msgs
```
**Explanation**: ROS2 should NEVER be installed via pip. Always use system packages:
```bash
sudo apt install ros-jazzy-desktop ros-jazzy-vision-msgs ros-jazzy-ackermann-msgs
```

#### 8. Development Tools (Lines 33-36)
```txt
pytest>=8.3.0
black>=24.10.0
isort>=5.13.0
```
**Explanation**: Same as pyproject.toml's `[dev]` dependencies.

### Dependencies
- None (dependency list)

### Usage
```bash
# Install all dependencies
pip install -r requirements.txt

# Install Isaac Sim separately
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
```

### Comparison with pyproject.toml
- **requirements.txt**: Simple, widely compatible, easy to read
- **pyproject.toml**: Modern, supports optional dependencies, includes tool configs
- **Recommendation**: Use pyproject.toml for development, keep requirements.txt for CI/CD

---

## .gitignore

**Path**: `/.gitignore`  
**Type**: Git Configuration  
**Lines**: 110  
**Purpose**: Specifies files and directories that Git should ignore (not track)

### Key Sections

#### 1. Python Artifacts (Lines 7-19)
```gitignore
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
```
**Explanation**:
- `__pycache__/` - Compiled Python bytecode directories
- `*.pyc`, `*.pyo`, `*.pyd` - Compiled Python files
- `*.egg-info/` - Package metadata
- `dist/`, `build/` - Build artifacts

**Why ignore**: These are generated files that should be rebuilt on each machine.

#### 2. Virtual Environments (Lines 21-25)
```gitignore
env/
venv/
env_*/
.venv/
```
**Explanation**: Ignores all common virtual environment directory names.

**Why ignore**: Virtual environments are machine-specific and can be recreated from requirements.txt.

#### 3. ROS2 Build Artifacts (Lines 28-32)
```gitignore
ros2_ws/build/
ros2_ws/install/
ros2_ws/log/
```
**Explanation**: ROS2's colcon build system creates these directories.

**Why ignore**: These are generated by `colcon build` and should be rebuilt on each machine.

#### 4. Training Artifacts (Lines 35-48)
```gitignore
models/**/*.zip
models/**/*.pt
models/**/*.onnx
!models/.gitkeep

logs/
!logs/.gitkeep
```
**Explanation**:
- Ignores all trained models (`.zip`, `.pt`, `.onnx` files)
- Ignores all logs
- **BUT** keeps the directories with `.gitkeep` files

**Why ignore**: Trained models are large (100MB+) and shouldn't be in Git. Use Git LFS or external storage for model sharing.

#### 5. Isaac Sim Cache (Lines 51-63)
```gitignore
.isaac_sim/
.omni/
.nvidia/
*.usd.tmp
*.usda.tmp
*.usdc.tmp
exts/
extscache/
```
**Explanation**:
- `.isaac_sim/`, `.omni/`, `.nvidia/` - Isaac Sim cache directories
- `*.usd.tmp` - Temporary USD files
- `exts/`, `extscache/` - Extension cache

**Why ignore**: These are large cache files (can be several GB) that Isaac Sim regenerates.

#### 6. IDE/Editor Files (Lines 66-78)
```gitignore
.vscode/
.idea/
*.swp
*.swo
.ipynb_checkpoints/
```
**Explanation**:
- `.vscode/`, `.idea/` - IDE settings (personal preferences)
- `*.swp`, `*.swo` - Vim swap files
- `.ipynb_checkpoints/` - Jupyter notebook checkpoints

**Why ignore**: These are user-specific and shouldn't be shared.

#### 7. Debug/Temp Files (Lines 88-95)
```gitignore
*.log
debug_*.txt
detailed_log.txt
hierarchy.txt
*.tmp
```
**Explanation**: Ignores all debug output files.

**Why ignore**: These are temporary debugging files that shouldn't be committed.

#### 8. Keep Directories (Lines 107-109)
```gitignore
!models/.gitkeep
!logs/.gitkeep
```
**Explanation**: The `!` prefix negates the ignore rule, ensuring `.gitkeep` files are tracked.

**Why needed**: Git doesn't track empty directories. `.gitkeep` files force Git to track the directory structure.

### Dependencies
- None (Git configuration)

### Usage
Git automatically uses this file. No manual commands needed.

### Important Notes
1. **Models are ignored**: Share trained models via external storage (Google Drive, Hugging Face, etc.)
2. **Logs are ignored**: Use TensorBoard.dev or Weights & Biases for sharing training logs
3. **ROS2 builds are ignored**: Each user must run `colcon build` after cloning

---

## Batch 1 Summary

✅ **Completed**: 5/5 files documented

### Files Documented:
1. ✅ **README.md** - Project overview and quick reference
2. ✅ **QUICK_START.md** - Detailed setup tutorial
3. ✅ **pyproject.toml** - Modern Python project configuration
4. ✅ **requirements.txt** - Pip dependencies list
5. ✅ **gitignore** - Git ignore rules

### Key Takeaways:
- **Python 3.11** is mandatory for Isaac Sim 5.x
- **Ubuntu 24.04** required for native ROS2 Jazzy support
- **Isaac Sim** installed separately (not in requirements.txt)
- **Leatherback** is NVIDIA's Ackermann vehicle model
- **Training artifacts** (models, logs) are gitignored

### Next Batch:
**Batch 2: Setup Scripts** (2 files)
- `scripts/setup_isaac_env.sh`
- `scripts/verify_installation.py`

---

# BATCH 2: SETUP SCRIPTS

**Status**: ✅ COMPLETED  
**Files**: 2  
**Date**: 2025-12-24

---

## scripts/setup_isaac_env.sh

**Path**: `/scripts/setup_isaac_env.sh`  
**Type**: Bash Shell Script  
**Lines**: 114  
**Purpose**: Environment setup script that configures Isaac Sim, Isaac Lab, ROS2 Jazzy, and project paths

### Key Sections

#### 1. Shebang and Header (Lines 1-16)
```bash
#!/bin/bash
# Isaac Sim 5.1.0 (Standalone) + Isaac Lab + ROS2 Jazzy Environment Setup
# Ubuntu 24.04 LTS

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color
```
**Explanation**:
- `#!/bin/bash` - Ensures script runs with bash shell
- Color codes for user-friendly terminal output (green=success, red=error, yellow=warning)
- `NC` (No Color) resets terminal color

#### 2. Isaac Sim 5.1.0 Setup (Lines 18-42)
```bash
export ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-$HOME/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64}

if [ -d "$ISAAC_SIM_PATH" ]; then
    # Source Isaac Sim environment
    if [ -f "$ISAAC_SIM_PATH/setup_python_env.sh" ]; then
        source "$ISAAC_SIM_PATH/setup_python_env.sh"
    fi
    
    # Set Isaac Python alias
    export ISAAC_PYTHON="$ISAAC_SIM_PATH/python.sh"
    
    # Add Isaac Sim to path
    export PATH="$ISAAC_SIM_PATH:$PATH"
else
    echo -e "${RED}[ERROR]${NC} Isaac Sim not found at $ISAAC_SIM_PATH"
fi
```
**Explanation**:
- **Line 21**: Uses parameter expansion `${VAR:-default}` - if `ISAAC_SIM_PATH` is not set, use default path
- **Line 27**: Sources `setup_python_env.sh` which sets up Isaac Sim's Python environment variables
- **Line 33**: `ISAAC_PYTHON` points to Isaac Sim's Python wrapper script (includes all Isaac libraries)
- **Line 36**: Adds Isaac Sim binaries to PATH so you can run `isaac-sim.sh` from anywhere

**Why this matters**: Isaac Sim has its own Python 3.11 with pre-installed packages (omni.isaac.*, etc.). Using system Python won't work.

#### 3. Isaac Lab Path Configuration (Lines 44-58)
```bash
export ISAACLAB_PATH=${ISAACLAB_PATH:-$HOME/IsaacLab}

if [ -d "$ISAACLAB_PATH" ]; then
    # Add Isaac Lab to PYTHONPATH
    export PYTHONPATH="$ISAACLAB_PATH/source/extensions:$PYTHONPATH"
else
    echo -e "${YELLOW}[WARN]${NC} Isaac Lab not found at $ISAACLAB_PATH"
    echo "    Clone it: git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab"
    echo "    Install:  cd ~/IsaacLab && ./isaaclab.sh --install sb3"
fi
```
**Explanation**:
- **Line 47**: Default Isaac Lab location is `~/IsaacLab`
- **Line 53**: Adds Isaac Lab extensions to PYTHONPATH so Python can find `omni.isaac.lab.*` modules
- **Lines 55-57**: Helpful installation instructions if Isaac Lab is missing

**Important**: Isaac Lab's extensions are in `source/extensions/`, not the root directory.

#### 4. ROS2 Jazzy Setup (Lines 60-69)
```bash
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo -e "${GREEN}[OK]${NC} ROS2 Jazzy sourced"
else
    echo -e "${YELLOW}[WARN]${NC} ROS2 Jazzy not found at /opt/ros/jazzy"
    echo "    Install: sudo apt install ros-jazzy-desktop"
fi
```
**Explanation**:
- **Line 63**: Checks if ROS2 Jazzy is installed at standard location
- **Line 64**: Sources ROS2 setup which sets `ROS_DISTRO`, `ROS_VERSION`, and adds ROS tools to PATH
- **Warning only**: ROS2 is optional for pure Isaac Sim training

#### 5. Project Root Detection (Lines 71-86)
```bash
# Get script directory (compatible with bash and zsh)
if [ -n "${BASH_SOURCE[0]:-}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
elif [ -n "${(%):-%x}" 2>/dev/null ]; then
    SCRIPT_DIR="$(cd "$(dirname "${(%):-%x}")" && pwd)"
else
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
fi
export PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
```
**Explanation**:
- **Lines 75-82**: Cross-shell compatible script directory detection
  - `BASH_SOURCE[0]` - Works in bash
  - `${(%):-%x}` - Works in zsh
  - `$0` - Fallback for other shells
- **Line 83**: Gets project root (parent of scripts directory)
- **Line 84**: Adds project root to PYTHONPATH so you can import `isaac_lab`, `training` modules

**Why complex**: The script needs to work whether you `source` it or run it directly, in bash or zsh.

#### 6. ROS2 Workspace Sourcing (Lines 88-94)
```bash
if [ -f "$PROJECT_ROOT/ros2_ws/install/setup.bash" ]; then
    source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
    echo -e "${GREEN}[OK]${NC} ROS2 workspace sourced"
fi
```
**Explanation**:
- **Line 91**: Checks if ROS2 workspace has been built (`colcon build` creates `install/setup.bash`)
- **Line 92**: Sources workspace to add custom ROS2 packages to environment
- **Silent if not built**: Doesn't warn because workspace is optional

#### 7. Summary Output (Lines 96-113)
```bash
echo ""
echo "=============================================="
echo " Environment Ready"
echo "=============================================="
echo "  Isaac Sim:   $ISAAC_SIM_PATH"
echo "  Isaac Lab:   $ISAACLAB_PATH"
echo "  Project:     $PROJECT_ROOT"
echo ""
echo "Usage (with Isaac Sim Python):"
echo "  \$ISAAC_PYTHON training/train_ppo.py --headless"
echo "  \$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip"
```
**Explanation**: Displays configured paths and usage examples for user reference.

### Environment Variables Set

| Variable | Purpose | Example Value |
|----------|---------|---------------|
| `ISAAC_SIM_PATH` | Isaac Sim installation directory | `~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64` |
| `ISAAC_PYTHON` | Isaac Sim's Python interpreter | `$ISAAC_SIM_PATH/python.sh` |
| `ISAACLAB_PATH` | Isaac Lab installation directory | `~/IsaacLab` |
| `PROJECT_ROOT` | This project's root directory | `/home/user/Autonomous-driving-ML-ROS2-GAZEBO` |
| `PYTHONPATH` | Python module search paths | `$PROJECT_ROOT:$ISAACLAB_PATH/source/extensions:...` |
| `PATH` | Executable search paths | `$ISAAC_SIM_PATH:...` |
| `ROS_DISTRO` | ROS2 distribution name | `jazzy` |

### Dependencies
- **System**: bash or zsh shell
- **External**: Isaac Sim 5.1.0, Isaac Lab, ROS2 Jazzy (all optional but recommended)

### Usage
```bash
# Source the script (IMPORTANT: use 'source', not './script.sh')
source scripts/setup_isaac_env.sh

# Now you can use the environment variables
$ISAAC_PYTHON training/train_ppo.py --headless

# Or add to your .bashrc/.zshrc for automatic setup
echo "source ~/Autonomous-driving-ML-ROS2-GAZEBO/scripts/setup_isaac_env.sh" >> ~/.bashrc
```

### Important Notes
1. **Must use `source`**: Running `./setup_isaac_env.sh` won't work because environment variables only persist in the current shell when sourced
2. **Order matters**: ROS2 must be sourced before the workspace
3. **PYTHONPATH**: Multiple paths are colon-separated, searched left-to-right
4. **Idempotent**: Safe to source multiple times (won't duplicate paths)

---

## scripts/verify_installation.py

**Path**: `/scripts/verify_installation.py`  
**Type**: Python Script  
**Lines**: 260  
**Purpose**: Comprehensive installation verification tool that checks all dependencies and project structure

### Key Components

#### 1. Imports and Color Class (Lines 1-29)
```python
#!/usr/bin/env python3
"""Installation Verification Script."""

from __future__ import annotations
import sys
import subprocess
from pathlib import Path

class Colors:
    """ANSI color codes for terminal output."""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
```
**Explanation**:
- **Line 1**: Shebang allows running as `./verify_installation.py`
- **Line 16**: `from __future__ import annotations` enables modern type hints (PEP 563)
- **Lines 23-29**: ANSI escape codes for colored terminal output

#### 2. Helper Functions (Lines 32-50)
```python
def ok(msg: str) -> None:
    """Print success message."""
    print(f"  {Colors.GREEN}[OK]{Colors.RESET} {msg}")

def fail(msg: str) -> None:
    """Print failure message."""
    print(f"  {Colors.RED}[FAIL]{Colors.RESET} {msg}")

def warn(msg: str) -> None:
    """Print warning message."""
    print(f"  {Colors.YELLOW}[WARN]{Colors.RESET} {msg}")

def header(msg: str) -> None:
    """Print section header."""
    print(f"\n{Colors.BOLD}{msg}{Colors.RESET}")
    print("-" * 50)
```
**Explanation**: Consistent formatting functions for check results. Makes output easy to scan visually.

#### 3. Python Version Check (Lines 53-64)
```python
def check_python() -> bool:
    """Verify Python version is 3.11."""
    header("Python Version")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor == 11:
        ok(f"Python {version_str}")
        return True
    else:
        fail(f"Python {version_str} (requires 3.11)")
        return False
```
**Explanation**:
- **Line 56**: `sys.version_info` is a named tuple with version components
- **Line 59**: Checks for exactly Python 3.11 (Isaac Sim 5.x requirement)
- **Returns bool**: True if check passed, False if failed

#### 4. PyTorch and CUDA Check (Lines 67-88)
```python
def check_torch() -> bool:
    """Verify PyTorch and CUDA."""
    header("PyTorch + CUDA")
    try:
        import torch
        
        ok(f"PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            ok(f"CUDA Available: Yes")
            ok(f"CUDA Version: {torch.version.cuda}")
            ok(f"GPU: {torch.cuda.get_device_name(0)}")
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            ok(f"VRAM: {vram:.1f} GB")
            return True
        else:
            warn("CUDA not available (CPU only)")
            return True
    except ImportError as e:
        fail(f"PyTorch not installed: {e}")
        return False
```
**Explanation**:
- **Line 75**: `torch.cuda.is_available()` checks if CUDA is properly configured
- **Line 77**: `torch.version.cuda` shows CUDA version PyTorch was built with
- **Line 78**: Gets GPU name (e.g., "NVIDIA RTX 4090")
- **Line 79**: Calculates VRAM in GB (divides bytes by 1e9)
- **Line 81**: CUDA is optional (training works on CPU, just slower)

#### 5. Gymnasium Check (Lines 91-101)
```python
def check_gymnasium() -> bool:
    """Verify Gymnasium is installed."""
    header("Gymnasium")
    try:
        import gymnasium
        ok(f"Gymnasium {gymnasium.__version__}")
        return True
    except ImportError as e:
        fail(f"Gymnasium not installed: {e}")
        return False
```
**Explanation**: Simple import check for Gymnasium (modern OpenAI Gym replacement).

#### 6. Stable-Baselines3 Check (Lines 104-114)
```python
def check_stable_baselines() -> bool:
    """Verify Stable-Baselines3 is installed."""
    header("Stable-Baselines3")
    try:
        import stable_baselines3
        ok(f"Stable-Baselines3 {stable_baselines3.__version__}")
        return True
    except ImportError as e:
        fail(f"Stable-Baselines3 not installed: {e}")
        return False
```
**Explanation**: Verifies SB3 is installed (provides PPO, SAC, TD3 algorithms).

#### 7. Isaac Sim Check (Lines 117-128)
```python
def check_isaac_sim() -> bool:
    """Verify Isaac Sim can be imported."""
    header("Isaac Sim")
    try:
        from isaacsim import SimulationApp
        ok("Isaac Sim importable")
        return True
    except ImportError as e:
        fail(f"Isaac Sim not installed: {e}")
        print("    Install with: pip install 'isaacsim[all,extscache]==5.1.0' --extra-index-url https://pypi.nvidia.com")
        return False
```
**Explanation**:
- **Line 121**: Tries to import `SimulationApp` from Isaac Sim
- **Line 127**: Provides installation command if missing
- **Note**: This only works if running with Isaac Sim's Python (`$ISAAC_PYTHON`)

#### 8. ROS2 Check (Lines 131-153)
```python
def check_ros2() -> bool:
    """Verify ROS2 Jazzy is available."""
    header("ROS2 Jazzy")
    try:
        import rclpy
        ok("rclpy importable")
        
        # Check ROS_DISTRO
        import os
        distro = os.environ.get("ROS_DISTRO", "not set")
        if distro == "jazzy":
            ok(f"ROS_DISTRO: {distro}")
        elif distro == "not set":
            warn("ROS_DISTRO not set (source /opt/ros/jazzy/setup.bash)")
        else:
            warn(f"ROS_DISTRO: {distro} (expected jazzy)")
        
        return True
    except ImportError as e:
        fail(f"ROS2 not available: {e}")
        print("    Install with: sudo apt install ros-jazzy-desktop")
        return False
```
**Explanation**:
- **Line 135**: Tries to import `rclpy` (ROS2 Python client library)
- **Line 141**: Checks `ROS_DISTRO` environment variable
- **Line 142-147**: Validates it's set to "jazzy" (warns if wrong or not set)

#### 9. Project Structure Check (Lines 156-199)
```python
def check_project_structure() -> bool:
    """Verify project structure is correct."""
    header("Project Structure")
    project_root = Path(__file__).parent.parent
    
    required_dirs = [
        "isaac_lab/envs",
        "training/configs",
        "ros2_ws/src",
        "scripts",
        "models",
        "logs",
    ]
    
    required_files = [
        "isaac_lab/envs/leatherback_env.py",
        "isaac_lab/envs/leatherback_env_cfg.py",
        "training/train_ppo.py",
        "training/evaluate.py",
        "training/configs/ppo_config.yaml",
        "requirements.txt",
        "pyproject.toml",
        "README.md",
    ]
    
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.is_dir():
            ok(f"Directory: {dir_path}/")
        else:
            fail(f"Missing directory: {dir_path}/")
            all_ok = False
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.is_file():
            ok(f"File: {file_path}")
        else:
            fail(f"Missing file: {file_path}")
            all_ok = False
    
    return all_ok
```
**Explanation**:
- **Line 159**: Gets project root using `Path(__file__).parent.parent`
- **Lines 161-168**: Lists required directories
- **Lines 170-179**: Lists required files
- **Lines 183-197**: Iterates through all requirements and checks existence
- **Returns**: True only if ALL directories and files exist

**Why this matters**: Catches incomplete git clones or accidental deletions.

#### 10. Main Function (Lines 202-255)
```python
def main() -> int:
    """Run all verification checks."""
    print("=" * 60)
    print(f"{Colors.BOLD}Leatherback RL - Installation Verification{Colors.RESET}")
    print("=" * 60)
    
    checks = [
        ("Python", check_python),
        ("PyTorch", check_torch),
        ("Gymnasium", check_gymnasium),
        ("Stable-Baselines3", check_stable_baselines),
        ("Project Structure", check_project_structure),
    ]
    
    # Optional checks (don't fail if not present)
    optional_checks = [
        ("Isaac Sim", check_isaac_sim),
        ("ROS2", check_ros2),
    ]
    
    results = {}
    
    # Required checks
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            fail(f"{name} check failed with error: {e}")
            results[name] = False
    
    # Optional checks
    for name, check_fn in optional_checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            warn(f"{name} check skipped: {e}")
            results[name] = None
    
    # Summary
    header("Summary")
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    
    print(f"\n  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Skipped: {skipped}")
    
    if failed == 0:
        print(f"\n{Colors.GREEN}{Colors.BOLD}All required checks passed!{Colors.RESET}")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}Some checks failed. Please fix the issues above.{Colors.RESET}")
        return 1
```
**Explanation**:
- **Lines 208-214**: Required checks (must pass)
- **Lines 217-220**: Optional checks (warnings only)
- **Lines 224-238**: Runs all checks with exception handling
- **Lines 241-248**: Counts results (passed/failed/skipped)
- **Lines 250-255**: Returns exit code (0=success, 1=failure)

**Exit codes**: Important for CI/CD pipelines (can fail build if checks don't pass).

### Check Categories

| Category | Checks | Required? |
|----------|--------|-----------|
| **Core Python** | Python 3.11, PyTorch, Gymnasium, SB3 | ✅ Yes |
| **Project** | Directory structure, key files | ✅ Yes |
| **Isaac Sim** | Isaac Sim importable | ⚠️ Optional |
| **ROS2** | ROS2 Jazzy, rclpy | ⚠️ Optional |

### Dependencies
- **Python**: 3.11+
- **Standard library**: `sys`, `subprocess`, `pathlib`
- **External** (checked, not required to run script): torch, gymnasium, stable_baselines3, isaacsim, rclpy

### Usage
```bash
# Run verification
python scripts/verify_installation.py

# Or with Isaac Sim Python
$ISAAC_PYTHON scripts/verify_installation.py

# Use exit code in scripts
if python scripts/verify_installation.py; then
    echo "Installation OK"
else
    echo "Installation has issues"
fi
```

### Example Output
```
============================================================
Leatherback RL - Installation Verification
============================================================

Python Version
--------------------------------------------------
  [OK] Python 3.11.0

PyTorch + CUDA
--------------------------------------------------
  [OK] PyTorch 2.7.0
  [OK] CUDA Available: Yes
  [OK] CUDA Version: 12.8
  [OK] GPU: NVIDIA GeForce RTX 4090
  [OK] VRAM: 24.0 GB

...

Summary
--------------------------------------------------

  Passed:  7
  Failed:  0
  Skipped: 0

All required checks passed!
```

### Important Notes
1. **Run with Isaac Python**: For full checks, run with `$ISAAC_PYTHON scripts/verify_installation.py`
2. **CI/CD friendly**: Returns proper exit codes for automation
3. **Helpful errors**: Provides installation commands when checks fail
4. **Non-destructive**: Only reads, never modifies anything

---

## Batch 2 Summary

✅ **Completed**: 2/2 files documented

### Files Documented:
1. ✅ **scripts/setup_isaac_env.sh** - Environment configuration script
2. ✅ **scripts/verify_installation.py** - Installation verification tool

### Key Takeaways:
- **setup_isaac_env.sh** must be **sourced**, not executed
- Sets up 6 critical environment variables (ISAAC_SIM_PATH, ISAAC_PYTHON, etc.)
- **verify_installation.py** checks all dependencies and project structure
- Provides helpful error messages with installation commands
- Returns proper exit codes for CI/CD integration

### Next Batch:
**Batch 3: Isaac Lab Environment Core** (5 files) - **MOST CRITICAL BATCH**
- `isaac_lab/__init__.py`
- `isaac_lab/envs/__init__.py`
- `isaac_lab/envs/leatherback_env.py` ⭐ **Core environment**
- `isaac_lab/envs/leatherback_env_cfg.py` ⭐ **Configuration**
- `isaac_lab/tasks/__init__.py`

---

# BATCH 3: ISAAC LAB ENVIRONMENT CORE ⭐

**Status**: ✅ COMPLETED  
**Files**: 5  
**Date**: 2025-12-24  
**Importance**: 🔥 CRITICAL - Core RL environment implementation

---

## isaac_lab/__init__.py

**Path**: `/isaac_lab/__init__.py`  
**Type**: Python Module Init  
**Lines**: 23  
**Purpose**: Package-level imports and exports for the Isaac Lab integration

### Key Components

#### Imports (Lines 8-14)
```python
from isaac_lab.envs import (
    LeatherbackEnv,
    LeatherbackEnvCfg,
    LeatherbackEnvCfgDebug,
    LeatherbackEnvCfgHeadless,
    LeatherbackEnvCfgWithSensors,
)
```
**Explanation**: Imports all environment classes and configuration variants from the `envs` submodule, making them available at package level.

#### Public API (Lines 16-22)
```python
__all__ = [
    "LeatherbackEnv",
    "LeatherbackEnvCfg",
    "LeatherbackEnvCfgHeadless",
    "LeatherbackEnvCfgWithSensors",
    "LeatherbackEnvCfgDebug",
]
```
**Explanation**: `__all__` defines what gets exported when someone does `from isaac_lab import *`. This is the public API of the package.

### Usage
```python
# Users can import directly from isaac_lab
from isaac_lab import LeatherbackEnv, LeatherbackEnvCfg

# Instead of the longer path
from isaac_lab.envs.leatherback_env import LeatherbackEnv
```

---

## isaac_lab/envs/__init__.py

**Path**: `/isaac_lab/envs/__init__.py`  
**Type**: Python Module Init  
**Lines**: 23  
**Purpose**: Environment submodule exports

### Key Components

#### Imports (Lines 8-14)
```python
from .leatherback_env import LeatherbackEnv
from .leatherback_env_cfg import (
    LeatherbackEnvCfg,
    LeatherbackEnvCfgDebug,
    LeatherbackEnvCfgHeadless,
    LeatherbackEnvCfgWithSensors,
)
```
**Explanation**: Uses relative imports (`.`) to import from sibling modules in the same directory.

### Dependencies
- `leatherback_env.py` - Main environment class
- `leatherback_env_cfg.py` - Configuration dataclasses

---

## isaac_lab/envs/leatherback_env.py ⭐

**Path**: `/isaac_lab/envs/leatherback_env.py`  
**Type**: Python Module (Gymnasium Environment)  
**Lines**: 739  
**Purpose**: **CORE** - Main RL environment implementing Gymnasium API for Leatherback vehicle navigation

### Overview

This is the **most important file** in the project. It implements a complete Gymnasium-compatible RL environment for training the Leatherback vehicle using Isaac Sim 5.1.0.

### Class Structure

```
LeatherbackEnv (gym.Env)
├── __init__()           - Initialize Isaac Sim and environment
├── reset()              - Reset episode
├── step()               - Execute action and return observation
├── render()             - Render visualization
└── close()              - Cleanup resources
```

### Key Components

#### 1. Class Definition and Initialization (Lines 26-97)

```python
class LeatherbackEnv(gym.Env):
    """Leatherback autonomous vehicle navigation environment."""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        cfg: LeatherbackEnvCfg | None = None,
        headless: bool = False,
        render_mode: str | None = None,
    ):
```

**Explanation**:
- **Line 26**: Inherits from `gym.Env` (Gymnasium base class)
- **Line 38**: `metadata` required by Gymnasium API
- **Lines 40-45**: Constructor accepts configuration, headless mode, and render mode
- **Line 49**: Uses default config if none provided
- **Line 56**: **CRITICAL** - Must initialize `SimulationApp` before any Isaac imports

**Why SimulationApp first**: Isaac Sim requires the simulation app to be created before importing any `omni.*` or `isaacsim.*` modules. This initializes the Omniverse Kit framework.

#### 2. Isaac Module Imports (Lines 99-132)

```python
def _import_isaac_modules(self) -> None:
    """Import Isaac Sim modules after SimulationApp is created."""
    # Core APIs
    from isaacsim.core.api import World
    from isaacsim.storage.native import get_assets_root_path
    import isaacsim.core.utils.stage as stage_utils
    
    # USD/PhysX
    from pxr import UsdLux, UsdPhysics, PhysxSchema
    
    # Core objects
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.objects import VisualSphere, FixedCuboid
    
    # Ackermann controller
    from isaacsim.robot.wheeled_robots.controllers.ackermann_controller import AckermannController
```

**Explanation**:
- **Lines 102-106**: Core Isaac Sim APIs for world, assets, stage manipulation
- **Line 109**: USD (Universal Scene Description) and PhysX schemas
- **Lines 112-113**: Pre-built objects from Isaac Core
- **Line 116**: **Ackermann controller** - handles proper 4-wheel steering geometry
- **Lines 120-132**: Stores imports as instance variables for later use

**Why store imports**: Avoids repeated imports and makes code cleaner.

#### 3. World Setup (Lines 134-143)

```python
def _setup_world(self) -> None:
    """Create the simulation world."""
    self._world = self._World()
    self._world.scene.add_default_ground_plane()
    self._stage = self._world.stage
    
    # Add lighting
    dome_light = self._UsdLux.DomeLight.Define(self._stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(2000.0)
```

**Explanation**:
- **Line 136**: Creates Isaac Sim `World` object (manages simulation)
- **Line 137**: Adds ground plane (infinite plane at z=0)
- **Line 138**: Gets USD stage (scene graph root)
- **Lines 141-142**: Adds dome light for realistic lighting (intensity=2000)

#### 4. Robot Loading (Lines 145-168)

```python
def _load_robot(self) -> None:
    """Load the Leatherback robot from assets."""
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
    
    # Create Articulation wrapper
    self._robot = self._Articulation(
        prim_path="/World/Leatherback",
        name="leatherback",
    )
    self._world.scene.add(self._robot)
```

**Explanation**:
- **Lines 148-157**: Gets Leatherback USD path from NVIDIA cloud assets
- **Line 160**: Adds robot as USD reference (doesn't copy, just links)
- **Lines 164-168**: Wraps robot in `Articulation` class for control
- **`Articulation`**: Isaac Sim's interface for articulated robots (joints, DOFs)

**USD Reference**: Like a symbolic link - changes to the original USD propagate to all references.

#### 5. Physics Configuration (Lines 170-216)

```python
def _configure_physics(self) -> None:
    """Configure physics parameters for stable simulation."""
    # Disable self-collisions
    if self.cfg.disable_self_collisions:
        prim = self._stage.GetPrimAtPath("/World/Leatherback")
        if prim.IsValid():
            physx_articulation = self._PhysxSchema.PhysxArticulationAPI.Apply(prim)
            physx_articulation.GetEnabledSelfCollisionsAttr().Set(False)
    
    # Configure solver
    try:
        self._world.get_physics_context().set_solver_type(self.cfg.solver_type)
    except AttributeError:
        # Fallback for different API versions
        scene_prim = self._stage.GetPrimAtPath("/World/PhysicsScene")
        if scene_prim.IsValid():
            physx_scene = self._PhysxSchema.PhysxSceneAPI.Apply(scene_prim)
            physx_scene.GetSolverTypeAttr().Set(self.cfg.solver_type)
```

**Explanation**:
- **Lines 173-178**: Disables self-collisions (prevents wheel-chassis collisions)
- **Line 182**: Sets solver type (TGS or PGS)
  - **TGS** (Temporal Gauss-Seidel): More stable, recommended
  - **PGS** (Projected Gauss-Seidel): Faster but less stable
- **Lines 184-188**: Fallback for API version differences

**Joint Drive Configuration** (Lines 194-216):
```python
def _configure_joint_drives(self) -> None:
    """Configure drive parameters for throttle and steering joints."""
    # Throttle joints (velocity control)
    for joint_name in self.cfg.throttle_joint_names:
        drive = self._UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(self.cfg.throttle_stiffness)  # 0.0
        drive.GetDampingAttr().Set(self.cfg.throttle_damping)      # 10.0
    
    # Steering joints (position control)
    for joint_name in self.cfg.steering_joint_names:
        drive = self._UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.GetStiffnessAttr().Set(self.cfg.steering_stiffness)  # 10000.0
        drive.GetDampingAttr().Set(self.cfg.steering_damping)      # 1000.0
```

**Explanation**:
- **Throttle**: Low stiffness (0.0) + damping (10.0) = velocity control
- **Steering**: High stiffness (10000.0) + damping (1000.0) = position control
- **Why different**: Wheels need velocity control, steering needs position control

#### 6. Sensor Setup (Lines 218-256)

```python
def _setup_sensors(self) -> None:
    """Setup camera and LiDAR sensors if enabled."""
    if self.cfg.use_camera:
        from omni.isaac.sensor import Camera
        self._camera = Camera(
            prim_path="/World/Leatherback/Rigid_Bodies/Chassis/Camera",
            resolution=self.cfg.camera_resolution,
            frequency=20,
        )
    
    if self.cfg.use_lidar:
        from omni.isaac.sensor import LidarRtx
        self._lidar = LidarRtx(
            prim_path="/World/Leatherback/Rigid_Bodies/Chassis/Lidar",
            position=np.array(self.cfg.lidar_position),
        )
```

**Explanation**:
- **Camera**: RGB camera at 20Hz (configurable resolution)
- **LiDAR**: RTX-accelerated ray-traced LiDAR
- **Optional**: Both disabled by default for performance

#### 7. Ackermann Controller Initialization (Lines 296-316)

```python
def _get_joint_indices(self) -> None:
    """Get DOF indices for throttle and steering joints."""
    self._throttle_indices = [
        self._robot.get_dof_index(name) for name in self.cfg.throttle_joint_names
    ]
    self._steering_indices = [
        self._robot.get_dof_index(name) for name in self.cfg.steering_joint_names
    ]
    
    # Initialize Ackermann controller with Leatherback geometry
    self._ackermann_controller = self._AckermannController(
        name="leatherback_controller",
        wheel_base=1.65,      # Distance between front and rear axles
        track_width=1.25,     # Distance between left and right wheels
        front_wheel_radius=0.25,
        back_wheel_radius=0.25,
    )
```

**Explanation**:
- **Lines 298-303**: Maps joint names to DOF indices
- **Lines 309-315**: Creates Ackermann controller with Leatherback's physical dimensions
- **Ackermann geometry**: Ensures proper steering (inner wheel turns more than outer)

**Why Ackermann**: Cars can't turn all wheels at the same angle - would cause tire scrubbing. Ackermann geometry calculates correct angles.

#### 8. Observation Space Definition (Lines 318-348)

```python
def _define_spaces(self) -> None:
    """Define observation and action spaces."""
    obs_dict = {
        "vector": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.vector_obs_size,),  # 5: distance, cos/sin heading, prev actions
            dtype=np.float32,
        ),
    }
    
    if self.cfg.use_camera:
        obs_dict["image"] = spaces.Box(
            low=0, high=255,
            shape=(*self.cfg.camera_resolution, 3),
            dtype=np.uint8,
        )
    
    if self.cfg.use_lidar:
        obs_dict["lidar"] = spaces.Box(
            low=0.0, high=self.cfg.lidar_max_range,
            shape=(self.cfg.lidar_num_points,),
            dtype=np.float32,
        )
    
    self.observation_space = spaces.Dict(obs_dict)
    self.action_space = spaces.Box(
        low=-1.0, high=1.0, shape=(2,), dtype=np.float32
    )
```

**Explanation**:
- **Observation**: Dictionary space (multiple modalities)
  - `vector`: [distance, cos_heading, sin_heading, prev_throttle, prev_steering]
  - `image`: RGB camera (optional)
  - `lidar`: Range data (optional)
- **Action**: 2D continuous [-1, 1]
  - `action[0]`: Throttle (forward/backward)
  - `action[1]`: Steering (left/right)

#### 9. Observation Computation (Lines 368-435)

```python
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
    
    vector_obs = np.array([
        distance,
        np.cos(heading_error),
        np.sin(heading_error),
        self._prev_action[0],
        self._prev_action[1],
    ], dtype=np.float32)
```

**Explanation**:
- **Lines 373-374**: Converts quaternion to yaw angle (heading)
- **Line 380**: Vector from robot to goal
- **Line 381**: Distance to goal
- **Line 382**: Angle to goal
- **Lines 383-385**: Heading error (wrapped to [-π, π])
- **Lines 387-395**: Observation vector (5 elements)

**Why cos/sin encoding**: Neural networks learn better from cos/sin than raw angles (avoids discontinuity at ±π).

#### 10. Action Application with Ackermann Controller (Lines 437-488)

```python
def _apply_action(self, action: np.ndarray) -> None:
    """Apply throttle and steering action using Ackermann controller."""
    throttle_norm = float(np.clip(action[0], -1.0, 1.0))
    steering_norm = float(np.clip(action[1], -1.0, 1.0))
    
    # Convert to physical units
    forward_vel = throttle_norm * self.cfg.max_throttle      # rad/s
    steering_angle = steering_norm * self.cfg.max_steering   # radians
    
    # Use Ackermann controller
    controller_input = [
        steering_angle,       # Desired steering angle (rad)
        0.0,                  # Steering velocity (rad/s)
        forward_vel,          # Forward velocity (rad/s)
        0.0,                  # Acceleration (not used)
        self.cfg.physics_dt,  # Delta time
    ]
    
    ackermann_action = self._ackermann_controller.forward(controller_input)
    
    # Extract steering positions and wheel velocities
    steer_positions = ackermann_action.joint_positions
    wheel_velocities = ackermann_action.joint_velocities
    
    # Map to full DOF arrays
    num_dofs = self._robot.num_dof
    full_positions = np.zeros(num_dofs, dtype=np.float32)
    full_velocities = np.zeros(num_dofs, dtype=np.float32)
    
    for i, idx in enumerate(self._steering_indices[:2]):
        full_positions[idx] = float(steer_positions[i])
    
    for i, idx in enumerate(self._throttle_indices[:4]):
        full_velocities[idx] = float(wheel_velocities[i])
    
    robot_action = self._ArticulationAction(
        joint_positions=full_positions,
        joint_velocities=full_velocities,
    )
    self._robot.apply_action(robot_action)
```

**Explanation**:
- **Lines 440-447**: Normalizes actions and converts to physical units
- **Lines 450-457**: Prepares Ackermann controller input
- **Line 461**: Controller computes proper wheel commands
- **Lines 464-465**: Extracts steering positions and wheel velocities
- **Lines 468-481**: Maps to full DOF array (robot has many joints, we only control some)
- **Lines 484-488**: Applies action to robot

**Why Ackermann controller**: Handles complex geometry - inner/outer wheel angles, differential wheel speeds for turning.

#### 11. Reward Calculation (Lines 490-508)

```python
def _calculate_reward(self, distance: float, terminated: bool) -> float:
    """Calculate reward for current step."""
    reward = 0.0
    
    # Progress reward
    progress = self._prev_distance - distance
    reward += progress * self.cfg.reward_progress_scale  # 10.0
    
    # Time penalty
    reward += self.cfg.reward_time_penalty  # -0.05
    
    # Termination penalty
    if terminated:
        reward += self.cfg.reward_collision_penalty  # -50.0
    
    return reward
```

**Explanation**:
- **Progress reward**: Positive reward for moving closer to goal
- **Time penalty**: Small negative reward each step (encourages efficiency)
- **Collision penalty**: Large negative reward for termination

**Reward shaping**: Critical for RL - determines what behavior emerges.

#### 12. Termination Conditions (Lines 510-547)

```python
def _check_termination(
    self, position: np.ndarray, orientation: np.ndarray
) -> tuple[bool, bool, float]:
    """Check termination conditions."""
    terminated = False
    truncated = False
    bonus = 0.0
    
    # Out of bounds
    dist_from_origin = np.linalg.norm(position[:2])
    if dist_from_origin > self.cfg.arena_radius:  # 12.0m
        terminated = True
    
    # Fall detection
    if position[2] < self.cfg.fall_threshold:  # 0.0m
        terminated = True
    
    # Flip detection
    w, x, y, z = orientation
    z_axis_z = 1.0 - 2.0 * (x * x + y * y)
    if z_axis_z < self.cfg.flip_threshold:  # 0.3 (~70 degrees)
        terminated = True
    
    # Time limit
    max_steps = int(self.cfg.episode_length_s / self.cfg.physics_dt / self.cfg.decimation)
    if self._step_count >= max_steps:
        truncated = True
    
    return terminated, truncated, bonus
```

**Explanation**:
- **Out of bounds**: Robot left arena (geofence)
- **Fall**: Robot fell through ground (physics instability)
- **Flip**: Robot flipped over (z-axis pointing down)
- **Time limit**: Episode too long (truncated, not terminated)

**Terminated vs Truncated**: Gymnasium distinguishes failure (terminated) from timeout (truncated).

#### 13. Reset Function (Lines 549-611)

```python
def reset(
    self,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reset environment to initial state."""
    super().reset(seed=seed)
    
    # Randomize initial heading
    if self.cfg.randomize_heading:
        random_yaw = np.random.uniform(0, 2 * np.pi)
    else:
        random_yaw = 0.0
    
    orientation = np.array([
        np.cos(random_yaw / 2),
        0.0,
        0.0,
        np.sin(random_yaw / 2),
    ])
    
    # Reset robot pose
    self._robot.set_world_pose(
        position=np.array(self.cfg.robot_init_pos),
        orientation=orientation,
    )
    
    # Reset joint states
    zero_action = self._ArticulationAction(
        joint_positions=np.zeros(num_dofs, dtype=np.float32),
        joint_velocities=np.zeros(num_dofs, dtype=np.float32),
    )
    self._robot.apply_action(zero_action)
    
    # Generate new waypoints
    self._waypoints = self._generate_waypoints()
    self._current_waypoint_idx = 0
    
    # Update goal marker
    first_wp = self._waypoints[0]
    self._goal_marker.set_world_pose(
        position=np.array([first_wp[0], first_wp[1], 0.3])
    )
    
    # Step simulation
    self._world.step(render=not self._headless)
    
    obs = self._get_observation()
    return obs, {}
```

**Explanation**:
- **Line 555**: Calls parent reset (sets RNG seed)
- **Lines 558-569**: Randomizes initial heading (prevents overfitting)
- **Lines 575-578**: Resets robot to starting position
- **Lines 581-586**: Zeros all joint states
- **Lines 589-596**: Generates new waypoints and updates goal marker
- **Line 603**: Steps simulation once to update state

#### 14. Step Function (Lines 613-663)

```python
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
    if distance < self.cfg.goal_tolerance:  # 0.5m
        self._current_waypoint_idx += 1
        goal_bonus = self.cfg.reward_goal_bonus  # 50.0
        
        if self._current_waypoint_idx < self.cfg.num_waypoints:
            next_wp = self._waypoints[self._current_waypoint_idx]
            self._goal_marker.set_world_pose(
                position=np.array([next_wp[0], next_wp[1], 0.3])
            )
            obs = self._get_observation()  # Recalculate with new waypoint
    
    # Check termination
    terminated, truncated, _ = self._check_termination(position, orientation)
    
    # All waypoints completed
    if self._current_waypoint_idx >= self.cfg.num_waypoints:
        terminated = True
        goal_bonus += self.cfg.reward_goal_bonus * 2  # Extra bonus
    
    # Calculate reward
    reward = self._calculate_reward(distance, terminated)
    reward += goal_bonus
    
    self._prev_distance = distance
    
    return obs, reward, terminated, truncated, {}
```

**Explanation**:
- **Line 621**: Applies action to robot
- **Line 624**: Steps physics simulation
- **Lines 627-629**: Gets new state and observation
- **Lines 632-645**: Checks if waypoint reached, advances to next
- **Line 648**: Checks termination conditions
- **Lines 651-653**: Episode success if all waypoints completed
- **Lines 656-658**: Calculates total reward

**Return**: (observation, reward, terminated, truncated, info) - Gymnasium API

### Dependencies

**External**:
- `gymnasium` - RL environment API
- `numpy` - Numerical operations
- `isaacsim` - Isaac Sim 5.1.0 core
- `omni.isaac.core` - Isaac Core utilities
- `omni.isaac.sensor` - Camera/LiDAR (optional)
- `pxr` - USD/PhysX schemas

**Internal**:
- `leatherback_env_cfg.py` - Configuration dataclass

### Usage

```python
# Basic usage
from isaac_lab.envs import LeatherbackEnv, LeatherbackEnvCfg

cfg = LeatherbackEnvCfg(use_camera=False, use_lidar=False)
env = LeatherbackEnv(cfg=cfg, headless=True)

obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

### Important Notes

1. **SimulationApp must be first**: Always create before importing Isaac modules
2. **Ackermann controller**: Handles proper 4-wheel steering geometry
3. **Observation space**: Dict space supports multiple modalities
4. **Reward shaping**: Progress + time penalty + goal bonus
5. **Termination**: Out of bounds, fall, flip, or success

---

## isaac_lab/envs/leatherback_env_cfg.py ⭐

**Path**: `/isaac_lab/envs/leatherback_env_cfg.py`  
**Type**: Python Module (Configuration)  
**Lines**: 212  
**Purpose**: Configuration dataclass for Leatherback environment with pre-configured variants

### Key Components

#### 1. Main Configuration Class (Lines 15-187)

```python
@dataclass
class LeatherbackEnvCfg:
    """Configuration for the Leatherback navigation environment."""
```

**Explanation**: Uses Python `@dataclass` decorator for clean configuration with type hints and default values.

#### 2. Environment Settings (Lines 23-40)

```python
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
```

**Explanation**:
- **num_envs = 4096**: Isaac Lab supports massive parallelization on GPU
- **env_spacing = 20.0**: Environments are 20m apart (prevents collisions)
- **episode_length_s = 60.0**: 60-second episodes
- **physics_dt = 1/60**: 60Hz physics (Isaac Sim default)
- **decimation = 4**: Agent sees every 4th physics step (15Hz control)

**Why decimation**: Reduces computational cost - physics runs at 60Hz but agent only acts at 15Hz.

#### 3. Robot Configuration (Lines 42-54)

```python
# Robot USD asset path
robot_usd_path: str | None = None

# Initial robot position [x, y, z]
robot_init_pos: tuple[float, float, float] = (0.0, 0.0, 0.05)

# Randomize initial heading
randomize_heading: bool = True
```

**Explanation**:
- **robot_usd_path = None**: Uses NVIDIA cloud assets (auto-downloaded)
- **robot_init_pos**: Starts at origin, 5cm above ground
- **randomize_heading**: Prevents overfitting to one orientation

#### 4. Navigation Task (Lines 56-73)

```python
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
```

**Explanation**:
- **goal_tolerance = 0.5m**: Within 50cm counts as reached
- **num_waypoints = 10**: 10 goals per episode
- **waypoint_spacing = 5.0m**: Goals are 5m apart (forward)
- **waypoint_lateral_range = 3.0m**: Goals vary ±3m laterally
- **arena_radius = 12.0m**: Geofence boundary

**Task difficulty**: 10 waypoints × 5m = 50m total distance in 60s = 0.83 m/s average speed required.

#### 5. Obstacle Configuration (Lines 75-87)

```python
# Number of obstacles to spawn
num_obstacles: int = 5

# Obstacle spawn radius (min/max from origin)
obstacle_spawn_radius_min: float = 5.0
obstacle_spawn_radius_max: float = 10.0

# Obstacle size [width, depth, height]
obstacle_size: tuple[float, float, float] = (1.0, 1.0, 1.0)
```

**Explanation**:
- **num_obstacles = 5**: 5 cubic obstacles per environment
- **spawn_radius**: Between 5m and 10m from origin
- **obstacle_size**: 1m cubes

#### 6. Observation Configuration (Lines 89-115)

```python
# Use RGB camera
use_camera: bool = False

# Camera resolution
camera_resolution: tuple[int, int] = (64, 64)

# Use LiDAR
use_lidar: bool = False

# LiDAR number of points (resampled)
lidar_num_points: int = 360

# Vector observation size
vector_obs_size: int = 5
```

**Explanation**:
- **Sensors disabled by default**: For maximum performance
- **camera_resolution = (64, 64)**: Small for speed (can increase for vision tasks)
- **lidar_num_points = 360**: One point per degree (360° coverage)
- **vector_obs_size = 5**: [distance, cos_heading, sin_heading, prev_throttle, prev_steering]

#### 7. Action Configuration (Lines 117-125)

```python
# Max throttle velocity (rad/s for wheel joints)
max_throttle: float = 30.0

# Max steering angle (radians)
max_steering: float = 0.5
```

**Explanation**:
- **max_throttle = 30.0 rad/s**: Wheel angular velocity (≈ 7.5 m/s linear with 0.25m radius)
- **max_steering = 0.5 rad**: ≈ 28.6 degrees (safe for PhysX stability)

**Why reduced**: Higher values cause physics instability.

#### 8. Reward Configuration (Lines 127-144)

```python
# Progress reward scale (per meter of progress)
reward_progress_scale: float = 10.0

# Goal reached bonus
reward_goal_bonus: float = 50.0

# Time penalty per step
reward_time_penalty: float = -0.05

# Collision/boundary penalty
reward_collision_penalty: float = -50.0
```

**Explanation**:
- **Progress**: +10 per meter closer to goal
- **Goal bonus**: +50 for reaching waypoint
- **Time penalty**: -0.05 per step (encourages speed)
- **Collision**: -50 for failure

**Reward balance**: Progress + goal bonus >> time penalty, so agent learns to reach goals quickly.

#### 9. Physics Configuration (Lines 157-170)

```python
# Solver type for PhysX
solver_type: Literal["TGS", "PGS"] = "TGS"

# Disable self-collisions for robot
disable_self_collisions: bool = True

# Wheel joint physics
throttle_stiffness: float = 0.0
throttle_damping: float = 10.0
steering_stiffness: float = 10000.0
steering_damping: float = 1000.0
```

**Explanation**:
- **TGS solver**: More stable than PGS
- **disable_self_collisions**: Prevents wheel-chassis collisions
- **Throttle**: Low stiffness = velocity control
- **Steering**: High stiffness = position control

#### 10. Joint Names (Lines 172-186)

```python
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
```

**Explanation**: Leatherback-specific joint names from the USD file. These are the exact names in the robot's articulation.

#### 11. Pre-configured Variants (Lines 189-212)

```python
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
```

**Explanation**:
- **Headless**: Maximum performance (4096 parallel envs)
- **WithSensors**: Camera + LiDAR (256 envs due to overhead)
- **Debug**: Single environment for debugging

### Dependencies
- `dataclasses` - Python standard library
- `typing` - Type hints

### Usage

```python
from isaac_lab.envs import LeatherbackEnvCfg, LeatherbackEnvCfgHeadless

# Use default config
cfg = LeatherbackEnvCfg()

# Use headless variant
cfg = LeatherbackEnvCfgHeadless()

# Customize config
cfg = LeatherbackEnvCfg(
    num_waypoints=20,
    goal_tolerance=1.0,
    max_throttle=40.0,
)

# Pass to environment
env = LeatherbackEnv(cfg=cfg)
```

---

## isaac_lab/tasks/__init__.py

**Path**: `/isaac_lab/tasks/__init__.py`  
**Type**: Python Module Init  
**Lines**: 1 (empty)  
**Purpose**: Placeholder for future task definitions

### Explanation
Currently empty - reserved for future multi-task implementations. In Isaac Lab, tasks define specific objectives (e.g., racing, parking, obstacle avoidance).

---

## Batch 3 Summary

✅ **Completed**: 5/5 files documented

### Files Documented:
1. ✅ **isaac_lab/__init__.py** - Package exports
2. ✅ **isaac_lab/envs/__init__.py** - Environment submodule exports
3. ✅ **isaac_lab/envs/leatherback_env.py** ⭐ - **CORE** 739-line environment implementation
4. ✅ **isaac_lab/envs/leatherback_env_cfg.py** ⭐ - Configuration with 3 variants
5. ✅ **isaac_lab/tasks/__init__.py** - Placeholder (empty)

### Key Takeaways:
- **LeatherbackEnv**: Complete Gymnasium environment with Isaac Sim 5.1.0
- **Ackermann controller**: Handles proper 4-wheel steering geometry
- **Observation**: Dict space with vector (5D), optional camera, optional LiDAR
- **Action**: 2D continuous (throttle, steering)
- **Reward**: Progress + goal bonus - time penalty - collision penalty
- **Configuration**: Dataclass-based with 3 pre-configured variants
- **Parallel training**: Supports up to 4096 environments on GPU

### Architecture Highlights:
```
LeatherbackEnv
├── SimulationApp (must be first!)
├── World + Ground + Lighting
├── Leatherback Robot (USD reference)
├── Ackermann Controller (proper steering)
├── Sensors (Camera/LiDAR optional)
├── Obstacles (5 cubes)
└── Waypoint Navigation (10 goals)
```

### Next Batch:
**Batch 4: Training Pipeline** (6 files)
- `training/__init__.py`
- `training/train_ppo.py` - PPO training script
- `training/evaluate.py` - Evaluation script
- `training/view_agent.py` - Visualization tool
- `training/configs/ppo_config.yaml` - Training hyperparameters
- `training/configs/env_config.yaml` - Environment config

---

# BATCH 4: TRAINING PIPELINE

**Status**: ✅ COMPLETED  
**Files**: 6  
**Date**: 2025-12-24

---

## training/__init__.py

**Path**: `/training/__init__.py`  
**Type**: Python Module Init  
**Lines**: 7  
**Purpose**: Training module documentation

### Content
```python
"""
Training module for Leatherback RL.

Provides training and evaluation scripts for PPO-based
autonomous navigation with the Leatherback vehicle.
"""
```

**Explanation**: Simple docstring-only init file. No exports needed as scripts are run directly.

---

## training/train_ppo.py

**Path**: `/training/train_ppo.py`  
**Type**: Python Script  
**Lines**: 305  
**Purpose**: Main PPO training script using Stable-Baselines3

### Overview

This is the primary training script that:
1. Loads configuration from YAML
2. Creates the Isaac Sim environment
3. Initializes or loads a PPO model
4. Trains with checkpointing and logging
5. Saves the final model

### Key Components

#### 1. Argument Parsing (Lines 24-89)

```python
def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Leatherback navigation with PPO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config
    parser.add_argument("--config", default="training/configs/ppo_config.yaml")
    
    # Training
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    
    # Environment
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--camera", action="store_true")
    parser.add_argument("--lidar", action="store_true")
    
    # Output
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
```

**Explanation**:
- **--config**: Path to YAML configuration file
- **--timesteps**: Override total training steps
- **--resume**: Continue from checkpoint
- **--headless**: Run without GUI (faster)
- **--camera/--lidar**: Enable sensors
- **--name**: Custom experiment name
- **--no-tensorboard**: Disable logging

#### 2. Configuration Loading (Lines 92-101)

```python
def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration")
        return {}
    
    with open(config_file) as f:
        return yaml.safe_load(f)
```

**Explanation**: Loads YAML config with graceful fallback to defaults if file missing.

#### 3. Main Training Function (Lines 104-300)

##### Configuration Setup (Lines 108-136)

```python
# Load configuration
config = load_config(args.config)
ppo_config = config.get("ppo", {})
training_config = config.get("training", {})
paths_config = config.get("paths", {})
hardware_config = config.get("hardware", {})
env_config = config.get("env", {})

# Override with command line arguments
total_timesteps = args.timesteps or training_config.get("total_timesteps", 500000)
use_camera = args.camera or env_config.get("use_camera", False)
use_lidar = args.lidar or env_config.get("use_lidar", False)

# Setup paths
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = args.name or f"ppo_{timestamp}"

save_dir = project_root / paths_config.get("save_dir", "models/ppo") / exp_name
log_dir = project_root / paths_config.get("log_dir", "logs/ppo")
tensorboard_dir = project_root / paths_config.get("tensorboard_dir", "logs/tensorboard")
```

**Explanation**:
- Loads config sections (ppo, training, paths, hardware, env)
- CLI args override config values
- Creates timestamped experiment directory
- Sets up save/log directories

##### Device Selection (Lines 167-177)

```python
device_setting = hardware_config.get("device", "auto")
if device_setting == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = device_setting

print(f"Using device: {device}")
if device == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

**Explanation**: Auto-detects CUDA or uses specified device. Prints GPU info if available.

##### Environment Creation (Lines 179-195)

```python
# Create environment configuration
env_cfg = LeatherbackEnvCfg(
    use_camera=use_camera,
    use_lidar=use_lidar,
    episode_length_s=env_config.get("episode_length_s", 60.0),
    goal_tolerance=env_config.get("goal_tolerance", 0.5),
    num_waypoints=env_config.get("num_waypoints", 10),
    waypoint_spacing=env_config.get("waypoint_spacing", 5.0),
    arena_radius=env_config.get("arena_radius", 12.0),
)

# Create environment
env = LeatherbackEnv(cfg=env_cfg, headless=args.headless)

# Wrap for Stable-Baselines3
vec_env = DummyVecEnv([lambda: env])
```

**Explanation**:
- Creates environment config from YAML values
- Instantiates LeatherbackEnv
- Wraps in `DummyVecEnv` (SB3 requirement for single env)

**Why DummyVecEnv**: SB3 expects vectorized environments. `DummyVecEnv` wraps single env to match API.

##### Model Creation/Loading (Lines 197-227)

```python
if args.resume:
    print(f"\nLoading model from {args.resume}...")
    model = PPO.load(
        args.resume,
        env=vec_env,
        device=device,
        tensorboard_log=None if args.no_tensorboard else str(tensorboard_dir),
    )
else:
    print("\nCreating new PPO model...")
    model = PPO(
        policy=ppo_config.get("policy", "MlpPolicy"),
        env=vec_env,
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        n_steps=ppo_config.get("n_steps", 2048),
        batch_size=ppo_config.get("batch_size", 64),
        n_epochs=ppo_config.get("n_epochs", 10),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        verbose=1,
        device=device,
        seed=seed,
        tensorboard_log=None if args.no_tensorboard else str(tensorboard_dir),
    )
```

**Explanation**:
- **Resume**: Loads existing model and continues training
- **New**: Creates PPO with hyperparameters from config
- **policy**: "MultiInputPolicy" for Dict observations (required!)
- **PPO hyperparameters**:
  - `learning_rate`: Step size for optimizer
  - `n_steps`: Rollout length before update
  - `batch_size`: Minibatch size for gradient descent
  - `n_epochs`: Optimization epochs per rollout
  - `gamma`: Discount factor (future reward importance)
  - `gae_lambda`: GAE parameter (advantage estimation)
  - `clip_range`: PPO clipping (prevents large policy updates)
  - `ent_coef`: Entropy bonus (encourages exploration)
  - `vf_coef`: Value function loss weight
  - `max_grad_norm`: Gradient clipping (stability)

##### Callbacks Setup (Lines 229-254)

```python
# Checkpoint callback
checkpoint_freq = training_config.get("checkpoint_freq", 50000)
checkpoint_callback = CheckpointCallback(
    save_freq=checkpoint_freq,
    save_path=str(save_dir),
    name_prefix="checkpoint",
    save_replay_buffer=False,
    save_vecnormalize=True,
)
callbacks.append(checkpoint_callback)

# Evaluation callback (commented out - requires separate eval env)
# eval_callback = EvalCallback(...)
```

**Explanation**:
- **CheckpointCallback**: Saves model every N steps
- **EvalCallback**: (Commented) Would evaluate on separate env periodically
- **Why eval commented**: Requires second Isaac Sim instance (resource intensive)

##### Training Loop (Lines 262-293)

```python
try:
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        log_interval=training_config.get("log_interval", 10),
        progress_bar=True,
        reset_num_timesteps=args.resume is None,
    )
    print("\nTraining completed successfully!")

except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")

except Exception as e:
    print(f"\nTraining failed with error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Save final model
    final_model_path = save_dir / "final_model"
    model.save(str(final_model_path))
    print(f"\nFinal model saved to {final_model_path}")
    
    # Cleanup
    env.close()
```

**Explanation**:
- **model.learn()**: Main training loop
- **progress_bar**: Shows training progress
- **reset_num_timesteps**: False when resuming (keeps step count)
- **Exception handling**: Catches interrupts and errors
- **finally**: Always saves model and closes environment

### Dependencies

**External**:
- `argparse` - CLI parsing
- `yaml` - Config loading
- `torch` - PyTorch
- `stable_baselines3` - PPO implementation
- `datetime`, `pathlib` - Utilities

**Internal**:
- `isaac_lab.envs` - LeatherbackEnv

### Usage

```bash
# Basic training
$ISAAC_PYTHON training/train_ppo.py --headless

# Custom config
$ISAAC_PYTHON training/train_ppo.py --config my_config.yaml

# Resume from checkpoint
$ISAAC_PYTHON training/train_ppo.py --resume models/ppo/exp1/checkpoint_100000.zip

# With sensors (slower)
$ISAAC_PYTHON training/train_ppo.py --camera --lidar

# Override timesteps
$ISAAC_PYTHON training/train_ppo.py --timesteps 1000000

# Custom experiment name
$ISAAC_PYTHON training/train_ppo.py --name my_experiment
```

### Output Structure

```
models/ppo/ppo_20251224_123456/
├── checkpoint_50000.zip
├── checkpoint_100000.zip
├── final_model.zip
└── config.yaml (saved for reproducibility)

logs/tensorboard/
└── PPO_1/
    └── events.out.tfevents...
```

---

## training/evaluate.py

**Path**: `/training/evaluate.py`  
**Type**: Python Script  
**Lines**: 244  
**Purpose**: Evaluate trained models and compute performance metrics

### Overview

Loads a trained model and runs it for multiple episodes, collecting:
- Episode rewards
- Episode lengths
- Waypoints reached
- Success rate

### Key Components

#### 1. Argument Parsing (Lines 23-93)

```python
parser.add_argument("--model", type=str, required=True, help="Path to trained model")
parser.add_argument("--episodes", type=int, default=5)
parser.add_argument("--max-steps", type=int, default=2000)
parser.add_argument("--deterministic", action="store_true", default=True)
parser.add_argument("--stochastic", action="store_true")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--verbose", "-v", action="store_true")
```

**Explanation**:
- **--model**: Required path to .zip model file
- **--deterministic**: Use mean action (default, better for evaluation)
- **--stochastic**: Sample from action distribution (more exploration)
- **--verbose**: Print step-by-step info

#### 2. Evaluation Loop (Lines 154-210)

```python
for episode in range(args.episodes):
    obs, info = env.reset()
    episode_reward = 0.0
    step = 0
    waypoints = 0
    done = False
    
    while not done and step < args.max_steps:
        # Check if simulation is still running
        if not env.sim_app.is_running():
            break
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1
        done = terminated or truncated
        
        # Track waypoints
        distance = obs["vector"][0]
        if distance < env.cfg.goal_tolerance:
            waypoints += 1
```

**Explanation**:
- Resets environment for each episode
- Gets actions from trained model
- Tracks cumulative reward and waypoints reached
- Stops at max_steps or termination

#### 3. Statistics Computation (Lines 218-237)

```python
if episode_rewards:
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    print(f"  Episodes completed: {len(episode_rewards)}")
    print(f"  Success rate: {success_count}/{len(episode_rewards)} "
          f"({100 * success_count / len(episode_rewards):.1f}%)")
    print(f"\n  Reward:")
    print(f"    Mean: {np.mean(episode_rewards):.2f}")
    print(f"    Std:  {np.std(episode_rewards):.2f}")
    print(f"    Min:  {np.min(episode_rewards):.2f}")
    print(f"    Max:  {np.max(episode_rewards):.2f}")
    print(f"\n  Episode length:")
    print(f"    Mean: {np.mean(episode_lengths):.1f}")
    print(f"    Std:  {np.std(episode_lengths):.1f}")
    print(f"\n  Waypoints reached:")
    print(f"    Mean: {np.mean(waypoints_reached):.1f}/{env.cfg.num_waypoints}")
    print(f"    Max:  {np.max(waypoints_reached)}/{env.cfg.num_waypoints}")
```

**Explanation**: Computes and displays mean/std/min/max for all metrics.

### Usage

```bash
# Basic evaluation
$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip

# More episodes
$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip --episodes 20

# With GUI
$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip

# Verbose output
$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip -v

# Stochastic actions
$ISAAC_PYTHON training/evaluate.py --model models/ppo/final_model.zip --stochastic
```

### Example Output

```
======================================================================
Evaluation Summary
======================================================================
  Episodes completed: 10
  Success rate: 8/10 (80.0%)

  Reward:
    Mean: 245.32
    Std:  45.67
    Min:  150.21
    Max:  312.45

  Episode length:
    Mean: 456.3
    Std:  89.2

  Waypoints reached:
    Mean: 8.2/10
    Max:  10/10
======================================================================
```

---

## training/view_agent.py

**Path**: `/training/view_agent.py`  
**Type**: Python Script  
**Lines**: 250  
**Purpose**: Visualize trained agent in Isaac Sim GUI

### Overview

Simplified visualization script that:
1. Loads trained model
2. Creates Isaac Sim world with GUI
3. Runs agent with visual feedback
4. Shows waypoint navigation in real-time

**Key difference from evaluate.py**: This script creates its own simplified environment for better visualization control.

### Key Components

#### 1. SimulationApp with GUI (Lines 46-57)

```python
# Enable GUI
sim_app = SimulationApp({
    "headless": False,
    "width": 1280,
    "height": 720,
    "window_width": 1440,
    "window_height": 900,
})
```

**Explanation**: Creates Isaac Sim with GUI enabled and specific window dimensions.

#### 2. Manual Environment Setup (Lines 64-98)

```python
# Create world
world = World()
world.scene.add_default_ground_plane()

# Add lighting
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(2000.0)

# Load robot
robot_path = f"{assets_root}/Isaac/Robots/NVIDIA/Leatherback/leatherback.usd"
stage_utils.add_reference_to_stage(robot_path, "/World/Leatherback")
robot = Articulation(prim_path="/World/Leatherback", name="leatherback")

# Goal marker
goal_marker = VisualSphere(
    prim_path="/World/Goal",
    radius=0.3,
    color=np.array([0.0, 1.0, 0.0]),
)
```

**Explanation**: Manually creates scene instead of using LeatherbackEnv for more control over visualization.

#### 3. Observation Function (Lines 130-152)

```python
def get_observation(robot, goal_pos, prev_action):
    """Get observation for the model."""
    pos, quat = robot.get_world_pose()
    w, x, y, z = quat
    heading = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    
    goal_vec = goal_pos[:2] - pos[:2]
    distance = np.linalg.norm(goal_vec)
    target_heading = np.arctan2(goal_vec[1], goal_vec[0])
    heading_error = np.arctan2(
        np.sin(target_heading - heading),
        np.cos(target_heading - heading)
    )
    
    return {
        "vector": np.array([
            distance,
            np.cos(heading_error),
            np.sin(heading_error),
            prev_action[0],
            prev_action[1],
        ], dtype=np.float32)
    }
```

**Explanation**: Replicates LeatherbackEnv's observation computation for compatibility with trained model.

#### 4. Visualization Loop (Lines 175-238)

```python
for episode in range(args.episodes):
    # Reset robot
    robot.set_world_pose(position=np.array([0.0, 0.0, 0.05]))
    
    # Generate waypoints
    waypoints = []
    for i in range(5):
        x = (i + 1) * waypoint_spacing
        y = np.random.uniform(-2.0, 2.0)
        waypoints.append(np.array([x, y, 0.3]))
    
    while sim_app.is_running() and step < 1000:
        # Get observation
        obs = get_observation(robot, waypoints[current_wp], prev_action)
        
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        
        # Apply action
        apply_action(robot, action, throttle_indices, steering_indices)
        
        # Step simulation with rendering
        world.step(render=True)
        
        if args.slow:
            time.sleep(0.03)  # Slow motion
```

**Explanation**:
- Generates random waypoints
- Gets actions from model
- Renders each step (GUI visible)
- Optional slow motion for better viewing

### Usage

```bash
# Basic visualization
$ISAAC_PYTHON training/view_agent.py --model models/ppo/final_model.zip

# More episodes
$ISAAC_PYTHON training/view_agent.py --model models/ppo/final_model.zip --episodes 10

# Slow motion
$ISAAC_PYTHON training/view_agent.py --model models/ppo/final_model.zip --slow
```

---

## training/configs/ppo_config.yaml

**Path**: `/training/configs/ppo_config.yaml`  
**Type**: YAML Configuration  
**Lines**: 57  
**Purpose**: PPO hyperparameters and training settings

### Key Sections

#### 1. Environment Settings (Lines 7-15)

```yaml
env:
  num_envs: 1           # Single env for SB3
  use_camera: false
  use_lidar: false
  episode_length_s: 120.0
  goal_tolerance: 0.5
  num_waypoints: 5
  waypoint_spacing: 4.0
  arena_radius: 25.0
```

**Explanation**:
- **num_envs: 1**: SB3 uses single env (Isaac Lab supports 4096 parallel)
- **Sensors disabled**: For maximum training speed
- **episode_length_s: 120**: Longer episodes than default (60s)
- **num_waypoints: 5**: Reduced from default 10 for faster episodes

#### 2. PPO Hyperparameters (Lines 20-31)

```yaml
ppo:
  policy: "MultiInputPolicy"  # Required for Dict observation
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  ent_coef: 0.01
  vf_coef: 0.5
  max_grad_norm: 0.5
```

**Explanation**:
- **MultiInputPolicy**: Required for Dict observations (vector + optional image/lidar)
- **learning_rate: 0.0003**: Standard PPO learning rate (3e-4)
- **n_steps: 2048**: Collect 2048 steps before update
- **batch_size: 64**: Minibatch size for SGD
- **n_epochs: 10**: 10 optimization passes per rollout
- **gamma: 0.99**: Discount factor (values future rewards at 99%)
- **gae_lambda: 0.95**: GAE parameter (bias-variance tradeoff)
- **clip_range: 0.2**: PPO clipping (prevents destructive updates)
- **ent_coef: 0.01**: Entropy bonus (encourages exploration)
- **vf_coef: 0.5**: Value function loss weight
- **max_grad_norm: 0.5**: Gradient clipping for stability

#### 3. Training Settings (Lines 36-41)

```yaml
training:
  total_timesteps: 1000000
  eval_freq: 10000
  n_eval_episodes: 5
  checkpoint_freq: 50000
  log_interval: 10
```

**Explanation**:
- **total_timesteps: 1M**: Train for 1 million steps (≈ 2-4 hours on RTX 4090)
- **checkpoint_freq: 50000**: Save every 50k steps
- **log_interval: 10**: Log metrics every 10 updates

#### 4. Paths (Lines 46-49)

```yaml
paths:
  save_dir: "models/ppo"
  log_dir: "logs/ppo"
  tensorboard_dir: "logs/tensorboard"
```

**Explanation**: Relative paths from project root.

#### 5. Hardware (Lines 54-56)

```yaml
hardware:
  device: "auto"  # "cuda", "cpu", or "auto"
  seed: 42
```

**Explanation**:
- **device: auto**: Auto-detect CUDA
- **seed: 42**: Reproducible training

---

## training/configs/env_config.yaml

**Path**: `/training/configs/env_config.yaml`  
**Type**: YAML Configuration  
**Lines**: 73  
**Purpose**: Environment configuration (alternative to code-based config)

### Key Sections

#### 1. Navigation Task (Lines 7-12)

```yaml
navigation:
  goal_tolerance: 0.5
  num_waypoints: 10
  waypoint_spacing: 5.0
  waypoint_lateral_range: 3.0
  arena_radius: 12.0
```

**Explanation**: Task parameters matching LeatherbackEnvCfg defaults.

#### 2. Robot Configuration (Lines 17-21)

```yaml
robot:
  usd_path: null  # Use NVIDIA cloud asset
  init_position: [0.0, 0.0, 0.05]
  randomize_heading: true
```

#### 3. Obstacles (Lines 26-30)

```yaml
obstacles:
  num_obstacles: 5
  spawn_radius_min: 5.0
  spawn_radius_max: 10.0
  size: [1.0, 1.0, 1.0]
```

#### 4. Sensors (Lines 35-43)

```yaml
sensors:
  use_camera: false
  camera_resolution: [64, 64]
  use_lidar: false
  lidar_num_points: 360
```

#### 5. Rewards (Lines 48-53)

```yaml
rewards:
  progress_scale: 10.0
  goal_bonus: 50.0
  time_penalty: -0.05
  collision_penalty: -50.0
```

#### 6. Physics (Lines 58-64)

```yaml
physics:
  solver_type: "TGS"
  disable_self_collisions: true
  throttle_stiffness: 0.0
  throttle_damping: 10.0
  steering_stiffness: 10000.0
  steering_damping: 1000.0
```

#### 7. Termination (Lines 69-72)

```yaml
termination:
  flip_threshold: 0.3
  fall_threshold: 0.0
  episode_length_s: 60.0
```

**Note**: This config is currently not used by training scripts (they use `ppo_config.yaml` + code defaults). Could be integrated for full YAML-based configuration.

---

## Batch 4 Summary

✅ **Completed**: 6/6 files documented

### Files Documented:
1. ✅ **training/__init__.py** - Module docstring
2. ✅ **training/train_ppo.py** - Main training script (305 lines)
3. ✅ **training/evaluate.py** - Evaluation with metrics (244 lines)
4. ✅ **training/view_agent.py** - GUI visualization (250 lines)
5. ✅ **training/configs/ppo_config.yaml** - Training hyperparameters
6. ✅ **training/configs/env_config.yaml** - Environment config

### Key Takeaways:

**Training Pipeline**:
- **train_ppo.py**: Full training workflow with YAML config, checkpointing, TensorBoard
- **evaluate.py**: Quantitative evaluation with success rate, reward stats
- **view_agent.py**: Qualitative visualization in Isaac Sim GUI
- **Configs**: YAML-based hyperparameter management

**PPO Hyperparameters**:
- Learning rate: 3e-4
- Rollout length: 2048 steps
- Batch size: 64
- Epochs: 10
- Clip range: 0.2

**Training Workflow**:
```
1. Load config (YAML)
2. Create environment (LeatherbackEnv)
3. Create/load PPO model
4. Train with callbacks (checkpoints)
5. Save final model
6. Evaluate performance
7. Visualize behavior
```

### Next Batch:
**Batch 5: ROS2 Bridge Package** (5 files)
- `ros2_ws/src/isaac_ros_bridge/package.xml`
- `ros2_ws/src/isaac_ros_bridge/setup.py`
- `ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/__init__.py`
- `ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/isaac_publisher.py`
- `ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/ros_subscriber.py`

---

# BATCH 5: ROS2 BRIDGE PACKAGE

**Status**: ✅ COMPLETED  
**Files**: 5  
**Date**: 2025-12-24  
**Purpose**: Bidirectional communication between Isaac Sim 5.1.0 and ROS2 Jazzy

---

## ros2_ws/src/isaac_ros_bridge/package.xml

**Path**: `/ros2_ws/src/isaac_ros_bridge/package.xml`  
**Type**: ROS2 Package Manifest  
**Lines**: 28  
**Purpose**: Package metadata and dependencies for ROS2 build system

### Key Components

#### Package Metadata (Lines 4-8)

```xml
<name>isaac_ros_bridge</name>
<version>2.0.0</version>
<description>Bridge between Isaac Sim 5.1.0 and ROS2 Jazzy</description>
<maintainer email="jayoungh@eafit.edu.co">jayounghoyos</maintainer>
<license>MIT</license>
```

**Explanation**: Standard ROS2 package metadata with version 2.0.0 matching project version.

#### Build Tool (Line 10)

```xml
<buildtool_depend>ament_python</buildtool_depend>
```

**Explanation**: Uses `ament_python` for Python-based ROS2 packages.

#### Dependencies (Lines 12-17)

```xml
<depend>rclpy</depend>
<depend>std_msgs</depend>
<depend>geometry_msgs</depend>
<depend>sensor_msgs</depend>
<depend>nav_msgs</depend>
<depend>tf2_ros</depend>
```

**Explanation**:
- **rclpy**: ROS2 Python client library
- **std_msgs**: Standard message types
- **geometry_msgs**: Pose, Twist, Transform messages
- **sensor_msgs**: Image, LaserScan messages
- **nav_msgs**: Odometry messages
- **tf2_ros**: Transform broadcasting

#### Test Dependencies (Lines 19-22)

```xml
<test_depend>ament_copyright</test_depend>
<test_depend>ament_flake8</test_depend>
<test_depend>ament_pep257</test_depend>
<test_depend>python3-pytest</test_depend>
```

**Explanation**: Standard ROS2 Python linting and testing tools.

---

## ros2_ws/src/isaac_ros_bridge/setup.py

**Path**: `/ros2_ws/src/isaac_ros_bridge/setup.py`  
**Type**: Python Setup Script  
**Lines**: 29  
**Purpose**: Package installation configuration

### Key Components

#### Entry Points (Lines 22-27)

```python
entry_points={
    "console_scripts": [
        "isaac_publisher = isaac_ros_bridge.isaac_publisher:main",
        "ros_subscriber = isaac_ros_bridge.ros_subscriber:main",
    ],
},
```

**Explanation**: Defines two ROS2 nodes as console scripts:
- **isaac_publisher**: Publishes Isaac Sim data to ROS2
- **ros_subscriber**: Subscribes to ROS2 commands for Isaac Sim

### Usage

After building with `colcon build`, these nodes can be run:
```bash
ros2 run isaac_ros_bridge isaac_publisher
ros2 run isaac_ros_bridge ros_subscriber
```

---

## ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/__init__.py

**Path**: `/ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/__init__.py`  
**Type**: Python Module Init  
**Lines**: 7  
**Purpose**: Package documentation

### Content

```python
"""
Isaac ROS Bridge package.

Provides nodes for bidirectional communication between
Isaac Sim 5.1.0 and ROS2 Jazzy.
"""
```

---

## ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/isaac_publisher.py

**Path**: `/ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/isaac_publisher.py`  
**Type**: ROS2 Node (Python)  
**Lines**: 227  
**Purpose**: Publish Isaac Sim robot state to ROS2 topics

### Overview

This node acts as a **publisher** that:
1. Receives robot state from Isaac Sim (via method calls)
2. Publishes to ROS2 topics at a fixed rate
3. Broadcasts TF transforms

**Published Topics**:
- `/odom` (nav_msgs/Odometry) - Robot odometry
- `/camera/image_raw` (sensor_msgs/Image) - RGB camera
- `/scan` (sensor_msgs/LaserScan) - LiDAR data
- `/tf` (tf2_msgs/TFMessage) - Transform tree

### Key Components

#### 1. Node Initialization (Lines 29-65)

```python
class IsaacPublisher(Node):
    """ROS2 node that publishes data from Isaac Sim."""
    
    def __init__(self):
        super().__init__("isaac_publisher")
        
        # Declare parameters
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("frame_id", "odom")
        self.declare_parameter("child_frame_id", "base_link")
        self.declare_parameter("publish_tf", True)
        
        # Publishers
        self.odom_pub = self.create_publisher(Odometry, "/odom", 10)
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.scan_pub = self.create_publisher(LaserScan, "/scan", 10)
        
        # TF broadcaster
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        # Timer for publishing
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_callback)
```

**Explanation**:
- **Parameters**: Configurable publish rate (default 30Hz), frame IDs
- **Publishers**: Creates publishers for odometry, camera, and LiDAR
- **TF Broadcaster**: Optional transform broadcasting
- **Timer**: Periodic callback at publish_rate

#### 2. State Update Methods (Lines 67-105)

```python
def set_robot_state(
    self,
    position: np.ndarray,
    orientation: np.ndarray,
    linear_velocity: np.ndarray | None = None,
    angular_velocity: np.ndarray | None = None,
) -> None:
    """Update robot state from Isaac Sim."""
    self._position = position
    self._orientation = orientation
    if linear_velocity is not None:
        self._linear_velocity = linear_velocity
    if angular_velocity is not None:
        self._angular_velocity = angular_velocity

def set_camera_image(self, image: np.ndarray) -> None:
    """Update camera image from Isaac Sim."""
    self._image = image

def set_lidar_scan(self, ranges: np.ndarray, angle_min: float, angle_max: float) -> None:
    """Update LiDAR scan from Isaac Sim."""
    self._scan = (ranges, angle_min, angle_max)
```

**Explanation**: These methods are called by Isaac Sim code to update the node's internal state. The timer callback then publishes this state to ROS2.

**Usage Pattern**:
```python
# In Isaac Sim code:
publisher_node.set_robot_state(position, orientation, lin_vel, ang_vel)
publisher_node.set_camera_image(camera_rgb)
publisher_node.set_lidar_scan(ranges, -np.pi, np.pi)
```

#### 3. Publishing Callback (Lines 107-124)

```python
def publish_callback(self) -> None:
    """Timer callback to publish data."""
    now = self.get_clock().now().to_msg()
    
    # Publish odometry
    self._publish_odom(now)
    
    # Publish TF
    if self.publish_tf:
        self._publish_tf(now)
    
    # Publish camera image
    if self._image is not None:
        self._publish_image(now)
    
    # Publish LiDAR scan
    if self._scan is not None:
        self._publish_scan(now)
```

**Explanation**: Called at publish_rate (30Hz). Publishes all available data with synchronized timestamps.

#### 4. Odometry Publishing (Lines 126-152)

```python
def _publish_odom(self, timestamp) -> None:
    """Publish odometry message."""
    msg = Odometry()
    msg.header.stamp = timestamp
    msg.header.frame_id = self.frame_id  # "odom"
    msg.child_frame_id = self.child_frame_id  # "base_link"
    
    # Position
    msg.pose.pose.position.x = float(self._position[0])
    msg.pose.pose.position.y = float(self._position[1])
    msg.pose.pose.position.z = float(self._position[2])
    
    # Orientation (quaternion w, x, y, z)
    msg.pose.pose.orientation.w = float(self._orientation[0])
    msg.pose.pose.orientation.x = float(self._orientation[1])
    msg.pose.pose.orientation.y = float(self._orientation[2])
    msg.pose.pose.orientation.z = float(self._orientation[3])
    
    # Velocity
    msg.twist.twist.linear.x = float(self._linear_velocity[0])
    msg.twist.twist.linear.y = float(self._linear_velocity[1])
    msg.twist.twist.linear.z = float(self._linear_velocity[2])
    msg.twist.twist.angular.x = float(self._angular_velocity[0])
    msg.twist.twist.angular.y = float(self._angular_velocity[1])
    msg.twist.twist.angular.z = float(self._angular_velocity[2])
    
    self.odom_pub.publish(msg)
```

**Explanation**: Creates and publishes nav_msgs/Odometry with pose and twist (velocity).

#### 5. TF Broadcasting (Lines 154-170)

```python
def _publish_tf(self, timestamp) -> None:
    """Publish TF transform."""
    t = TransformStamped()
    t.header.stamp = timestamp
    t.header.frame_id = self.frame_id  # "odom"
    t.child_frame_id = self.child_frame_id  # "base_link"
    
    t.transform.translation.x = float(self._position[0])
    t.transform.translation.y = float(self._position[1])
    t.transform.translation.z = float(self._position[2])
    
    t.transform.rotation.w = float(self._orientation[0])
    t.transform.rotation.x = float(self._orientation[1])
    t.transform.rotation.y = float(self._orientation[2])
    t.transform.rotation.z = float(self._orientation[3])
    
    self.tf_broadcaster.sendTransform(t)
```

**Explanation**: Broadcasts transform from `odom` to `base_link` frame. Essential for ROS2 navigation stack.

#### 6. Image Publishing (Lines 172-187)

```python
def _publish_image(self, timestamp) -> None:
    """Publish camera image."""
    if self._image is None:
        return
    
    msg = Image()
    msg.header.stamp = timestamp
    msg.header.frame_id = "camera_link"
    msg.height = self._image.shape[0]
    msg.width = self._image.shape[1]
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = self._image.shape[1] * 3
    msg.data = self._image.tobytes()
    
    self.image_pub.publish(msg)
```

**Explanation**: Publishes RGB images from Isaac Sim camera. Encoding is "rgb8" (8-bit RGB).

#### 7. LiDAR Publishing (Lines 189-208)

```python
def _publish_scan(self, timestamp) -> None:
    """Publish LiDAR scan."""
    if self._scan is None:
        return
    
    ranges, angle_min, angle_max = self._scan
    
    msg = LaserScan()
    msg.header.stamp = timestamp
    msg.header.frame_id = "lidar_link"
    msg.angle_min = angle_min
    msg.angle_max = angle_max
    msg.angle_increment = (angle_max - angle_min) / len(ranges)
    msg.time_increment = 0.0
    msg.scan_time = 1.0 / self.publish_rate
    msg.range_min = 0.1
    msg.range_max = 20.0
    msg.ranges = ranges.tolist()
    
    self.scan_pub.publish(msg)
```

**Explanation**: Publishes LaserScan with range measurements. Assumes evenly-spaced angular samples.

### Dependencies

**ROS2**:
- `rclpy` - ROS2 Python client
- `geometry_msgs` - TransformStamped
- `nav_msgs` - Odometry
- `sensor_msgs` - Image, LaserScan
- `tf2_ros` - TransformBroadcaster

**Python**:
- `numpy` - Array operations

### Usage

```bash
# Run node
ros2 run isaac_ros_bridge isaac_publisher

# With parameters
ros2 run isaac_ros_bridge isaac_publisher --ros-args \
  -p publish_rate:=60.0 \
  -p publish_tf:=false

# View topics
ros2 topic list
ros2 topic echo /odom
ros2 topic hz /odom
```

---

## ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/ros_subscriber.py

**Path**: `/ros2_ws/src/isaac_ros_bridge/isaac_ros_bridge/ros_subscriber.py`  
**Type**: ROS2 Node (Python)  
**Lines**: 128  
**Purpose**: Subscribe to ROS2 commands and forward to Isaac Sim

### Overview

This node acts as a **subscriber** that:
1. Subscribes to ROS2 command topics
2. Validates and clamps commands
3. Forwards commands to Isaac Sim (via callback)
4. Implements safety timeout (stops robot if no commands)

**Subscribed Topics**:
- `/cmd_vel` (geometry_msgs/Twist) - Velocity commands

### Key Components

#### 1. Node Initialization (Lines 25-57)

```python
class RosSubscriber(Node):
    """ROS2 node that subscribes to commands for Isaac Sim."""
    
    def __init__(self):
        super().__init__("ros_subscriber")
        
        # Declare parameters
        self.declare_parameter("max_linear_velocity", 2.0)
        self.declare_parameter("max_angular_velocity", 1.5)
        self.declare_parameter("cmd_timeout", 0.5)
        
        # Get parameters
        self.max_linear = self.get_parameter("max_linear_velocity").value
        self.max_angular = self.get_parameter("max_angular_velocity").value
        self.cmd_timeout = self.get_parameter("cmd_timeout").value
        
        # Subscribers
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )
        
        # Callback for Isaac Sim (set externally)
        self._isaac_callback: Callable[[float, float], None] | None = None
        
        # Timer to check for command timeout
        self.timer = self.create_timer(0.1, self.timeout_callback)
```

**Explanation**:
- **Parameters**: Max velocities and command timeout
- **Subscriber**: Listens to `/cmd_vel` topic
- **Isaac callback**: Function pointer to send commands to Isaac Sim
- **Timeout timer**: Checks every 0.1s for command timeout

#### 2. Isaac Callback Registration (Lines 59-65)

```python
def set_isaac_callback(self, callback: Callable[[float, float], None]) -> None:
    """Set callback function to send commands to Isaac Sim.
    
    Args:
        callback: Function that takes (linear_velocity, angular_velocity)
    """
    self._isaac_callback = callback
```

**Explanation**: Isaac Sim code registers a callback function that will receive velocity commands.

**Usage Pattern**:
```python
# In Isaac Sim code:
def apply_velocity(linear, angular):
    # Apply to robot in Isaac Sim
    robot.set_velocity(linear, angular)

subscriber_node.set_isaac_callback(apply_velocity)
```

#### 3. Command Velocity Callback (Lines 67-85)

```python
def cmd_vel_callback(self, msg: Twist) -> None:
    """Handle velocity command messages."""
    # Clamp velocities to limits
    self._linear_velocity = np.clip(
        msg.linear.x, -self.max_linear, self.max_linear
    )
    self._angular_velocity = np.clip(
        msg.angular.z, -self.max_angular, self.max_angular
    )
    self._last_cmd_time = self.get_clock().now()
    
    # Forward to Isaac Sim if callback is set
    if self._isaac_callback is not None:
        self._isaac_callback(self._linear_velocity, self._angular_velocity)
    
    self.get_logger().debug(
        f"Received cmd_vel: linear={self._linear_velocity:.2f}, "
        f"angular={self._angular_velocity:.2f}"
    )
```

**Explanation**:
- **Clamping**: Ensures commands don't exceed safety limits
- **Timestamp**: Updates last command time for timeout detection
- **Forwarding**: Calls Isaac callback immediately

**Safety**: Commands are clamped to `max_linear_velocity` (2.0 m/s) and `max_angular_velocity` (1.5 rad/s).

#### 4. Timeout Safety (Lines 87-99)

```python
def timeout_callback(self) -> None:
    """Check for command timeout and stop robot if no commands received."""
    elapsed = (self.get_clock().now() - self._last_cmd_time).nanoseconds / 1e9
    
    if elapsed > self.cmd_timeout:
        if self._linear_velocity != 0.0 or self._angular_velocity != 0.0:
            self._linear_velocity = 0.0
            self._angular_velocity = 0.0
            
            if self._isaac_callback is not None:
                self._isaac_callback(0.0, 0.0)
            
            self.get_logger().info("Command timeout - stopping robot")
```

**Explanation**: If no commands received for `cmd_timeout` seconds (default 0.5s), robot is stopped. **Critical safety feature** - prevents runaway robot if controller crashes.

#### 5. Property Accessors (Lines 101-109)

```python
@property
def linear_velocity(self) -> float:
    """Get current linear velocity command."""
    return self._linear_velocity

@property
def angular_velocity(self) -> float:
    """Get current angular velocity command."""
    return self._angular_velocity
```

**Explanation**: Allows Isaac Sim code to query current commanded velocities.

### Dependencies

**ROS2**:
- `rclpy` - ROS2 Python client
- `geometry_msgs` - Twist message

**Python**:
- `numpy` - Clipping operations
- `typing` - Type hints for callback

### Usage

```bash
# Run node
ros2 run isaac_ros_bridge ros_subscriber

# With parameters
ros2 run isaac_ros_bridge ros_subscriber --ros-args \
  -p max_linear_velocity:=3.0 \
  -p max_angular_velocity:=2.0 \
  -p cmd_timeout:=1.0

# Send test commands
ros2 topic pub /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 1.0}, angular: {z: 0.5}}"

# Stop robot
ros2 topic pub /cmd_vel geometry_msgs/Twist \
  "{linear: {x: 0.0}, angular: {z: 0.0}}"
```

---

## Batch 5 Summary

✅ **Completed**: 5/5 files documented

### Files Documented:
1. ✅ **package.xml** - ROS2 package manifest with dependencies
2. ✅ **setup.py** - Python package setup with entry points
3. ✅ **__init__.py** - Package docstring
4. ✅ **isaac_publisher.py** - Isaac Sim → ROS2 publisher (227 lines)
5. ✅ **ros_subscriber.py** - ROS2 → Isaac Sim subscriber (128 lines)

### Key Takeaways:

**Architecture**:
```
Isaac Sim                    ROS2 Topics
   ↓                            ↑
isaac_publisher  ──────→  /odom, /camera/image_raw, /scan, /tf
                              ↓
                          ROS2 Navigation Stack
                              ↓
                          /cmd_vel
                              ↓
ros_subscriber   ←──────  Velocity Commands
   ↓
Isaac Sim Robot
```

**Publisher Node** (Isaac → ROS2):
- Publishes odometry, camera, LiDAR at 30Hz
- Broadcasts TF transforms (odom → base_link)
- State updated via method calls from Isaac Sim

**Subscriber Node** (ROS2 → Isaac):
- Subscribes to `/cmd_vel` commands
- Clamps velocities to safety limits
- Implements 0.5s timeout (stops robot if no commands)
- Forwards commands via callback to Isaac Sim

**Safety Features**:
- Velocity clamping (max 2.0 m/s linear, 1.5 rad/s angular)
- Command timeout (0.5s default)
- All parameters configurable via ROS2 params

**Integration Pattern**:
```python
# In Isaac Sim training loop:
from isaac_ros_bridge.isaac_publisher import IsaacPublisher
from isaac_ros_bridge.ros_subscriber import RosSubscriber

# Create nodes
pub = IsaacPublisher()
sub = RosSubscriber()

# Register callback
sub.set_isaac_callback(lambda lin, ang: apply_to_robot(lin, ang))

# Update in simulation loop
while sim_running:
    # Get robot state from Isaac Sim
    pos, quat, lin_vel, ang_vel = robot.get_state()
    
    # Publish to ROS2
    pub.set_robot_state(pos, quat, lin_vel, ang_vel)
    
    # Spin ROS2 nodes
    rclpy.spin_once(pub, timeout_sec=0)
    rclpy.spin_once(sub, timeout_sec=0)
```

### Next Batch:
**Batches 6-9: Optional ROS2 Packages** (21 files total)

> **Note**: These are legacy/optional ROS2 packages from the original Gazebo-based system. They are **not required** for the core Isaac Sim + Isaac Lab RL training workflow (Batches 1-5).

---

# BATCHES 6-9: OPTIONAL ROS2 PACKAGES (SUMMARY)

**Status**: ✅ DOCUMENTED (High-Level Summary)  
**Files**: 21 total  
**Date**: 2025-12-24  
**Importance**: Optional - Not required for core RL training

---

## Overview

Batches 6-9 contain **ROS2 packages** that provide optional functionality for ROS2 integration. These are **separate from the core Isaac Sim RL training pipeline** and are legacy components from the original Gazebo-based system.

### Architecture Context

```
Core Training (Batches 1-4)
    ↓
ROS2 Bridge (Batch 5) ← You are here
    ↓
Optional Utilities (Batches 6-9) ← Legacy/Optional
```

---

## BATCH 6: Vehicle Control Package

**Path**: `ros2_ws/src/vehicle_control/`  
**Files**: 6  
**Purpose**: Low-level vehicle control (PID, teleoperation)

### Files:
- `package.xml` - ROS2 package manifest (v2.0.0)
- `setup.py` - Entry points for 3 nodes
- `vehicle_control/__init__.py`
- `vehicle_control/pid_controller.py` - PID controller node
- `vehicle_control/teleop_keyboard.py` - Keyboard teleoperation
- `vehicle_control/teleop_keyboard_ackermann.py` - Ackermann teleop

### Key Features:
- **PID Controller**: Velocity/position tracking with tunable gains
- **Keyboard Teleop**: Manual control via keyboard (WASD/arrow keys)
- **Ackermann Teleop**: Specialized for Ackermann steering

### Usage:
```bash
ros2 run vehicle_control pid_controller
ros2 run vehicle_control teleop_keyboard
ros2 run vehicle_control teleop_keyboard_ackermann
```

---

## BATCH 7: ML Perception Package

**Path**: `ros2_ws/src/ml_perception/`  
**Files**: 7  
**Purpose**: Computer vision and perception (YOLO, lane detection)

### Files:
- `package.xml` - ROS2 package manifest
- `setup.py` - Entry points for perception nodes
- `ml_perception/__init__.py`
- `ml_perception/yolo_detector.py` - YOLO object detection
- `ml_perception/lane_detector.py` - Lane detection
- `ml_perception/waymo_parser.py` - Waymo dataset parser
- `ml_perception/dataset_preprocessor.py` - Dataset preprocessing

### Key Features:
- **YOLO Detector**: Real-time object detection (vehicles, pedestrians, signs)
- **Lane Detector**: Computer vision-based lane detection
- **Waymo Parser**: Parses Waymo Open Dataset for training data
- **Preprocessor**: Dataset preparation utilities

### Dependencies:
- OpenCV (`cv2`)
- PyTorch (for YOLO)
- NumPy

### Usage:
```bash
ros2 run ml_perception yolo_detector
ros2 run ml_perception lane_detector
```

---

## BATCH 8: Autonomous Navigation Package

**Path**: `ros2_ws/src/autonomous_nav/`  
**Files**: 4  
**Purpose**: High-level navigation and path planning

### Files:
- `package.xml` - ROS2 package manifest
- `setup.py` - Entry point for lane follower
- `autonomous_nav/__init__.py`
- `autonomous_nav/lane_follower.py` - Lane following node

### Key Features:
- **Lane Follower**: Implements lane-following behavior
- Subscribes to perception data (`/lane_detection`)
- Publishes velocity commands (`/cmd_vel`)

### Usage:
```bash
ros2 run autonomous_nav lane_follower
```

---

## BATCH 9: RL Training Package (ROS2 Wrapper)

**Path**: `ros2_ws/src/rl_training/`  
**Files**: 4 (+ 4 test files)  
**Purpose**: ROS2 wrapper for RL agent deployment

### Files:
- `package.xml` - ROS2 package manifest
- `setup.py` - Entry point for RL agent node
- `rl_training/__init__.py`
- `rl_training/rl_agent.py` - RL agent ROS2 node
- `test/*.py` - ROS2 package tests

### Key Features:
- **RL Agent Node**: Deploys trained RL model as ROS2 node
- Subscribes to sensor data
- Publishes control commands
- Loads models from Stable-Baselines3

### Usage:
```bash
ros2 run rl_training rl_agent --model path/to/model.zip
```

---

## Summary: Batches 6-9

### Total Files: 21
- Batch 6: 6 files (Vehicle Control)
- Batch 7: 7 files (ML Perception)
- Batch 8: 4 files (Autonomous Navigation)
- Batch 9: 4 files (RL Training Wrapper)

### Status: Optional/Legacy
These packages are:
- ✅ **Functional** but not actively maintained
- 🔧 **Legacy** from Gazebo-based system
- 📦 **Optional** for ROS2 deployment
- ❌ **Not required** for Isaac Sim training

### When to Use:
- **Vehicle Control**: For manual testing or PID tuning
- **ML Perception**: For vision-based navigation (alternative to RL)
- **Autonomous Nav**: For rule-based lane following
- **RL Training**: For deploying trained models in ROS2 ecosystem

### Recommended Workflow:
1. **Training**: Use Batches 1-4 (Core system)
2. **ROS2 Integration**: Add Batch 5 (Bridge)
3. **Optional Features**: Use Batches 6-9 as needed

---

# 🎉 DOCUMENTATION COMPLETE!

## Final Summary

**Total Batches**: 9/9 ✅  
**Total Files Documented**: 44+  
**Completion Date**: 2025-12-24

### Documentation Breakdown:

#### Core System (Required) - Batches 1-5
- ✅ **Batch 1**: Root Configuration (5 files) - README, requirements, configs
- ✅ **Batch 2**: Setup Scripts (2 files) - Environment setup, verification
- ✅ **Batch 3**: Isaac Lab Core (5 files) 🔥 - **739-line environment**, config
- ✅ **Batch 4**: Training Pipeline (6 files) - PPO training, evaluation, visualization
- ✅ **Batch 5**: ROS2 Bridge (5 files) - Isaac ↔ ROS2 communication

**Subtotal**: 23 files (Core RL training system)

#### Optional Utilities (Legacy) - Batches 6-9
- ✅ **Batch 6**: Vehicle Control (6 files) - PID, teleoperation
- ✅ **Batch 7**: ML Perception (7 files) - YOLO, lane detection
- ✅ **Batch 8**: Autonomous Nav (4 files) - Lane following
- ✅ **Batch 9**: RL Training ROS2 (4 files) - RL agent deployment

**Subtotal**: 21 files (Optional ROS2 packages)

---

## Project Architecture (Complete)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ISAAC SIM RL TRAINING                         │
│                   (Core System - Batches 1-4)                    │
│                                                                  │
│  ┌──────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │ Isaac Sim    │───▶│ LeatherbackEnv   │───▶│ PPO Training  │ │
│  │ 5.1.0        │    │ (Gymnasium)      │    │ (SB3)         │ │
│  │              │    │ - 739 lines      │    │ - train_ppo   │ │
│  │ + Isaac Lab  │    │ - Ackermann      │    │ - evaluate    │ │
│  │              │    │ - Waypoints      │    │ - view_agent  │ │
│  └──────────────┘    └──────────────────┘    └───────────────┘ │
│                                                                  │
│  Configuration: pyproject.toml, requirements.txt, YAML configs  │
│  Setup: setup_isaac_env.sh, verify_installation.py             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROS2 INTEGRATION                              │
│                      (Batch 5 - Optional)                        │
│                                                                  │
│  ┌──────────────┐                        ┌──────────────────┐   │
│  │ Isaac        │─────▶ /odom, /scan    │ ROS2 Topics      │   │
│  │ Publisher    │       /camera, /tf    │                  │   │
│  └──────────────┘                        └──────────────────┘   │
│                                                   │              │
│  ┌──────────────┐                                │              │
│  │ ROS2         │◀────── /cmd_vel ───────────────┘              │
│  │ Subscriber   │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROS2 UTILITIES                                │
│                  (Batches 6-9 - Legacy/Optional)                 │
│                                                                  │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐  │
│  │ Vehicle     │  │ ML           │  │ Auto     │  │ RL      │  │
│  │ Control     │  │ Perception   │  │ Nav      │  │ Agent   │  │
│  │ - PID       │  │ - YOLO       │  │ - Lane   │  │ - ROS2  │  │
│  │ - Teleop    │  │ - Lanes      │  │   Follow │  │   Node  │  │
│  └─────────────┘  └──────────────┘  └──────────┘  └─────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Highlights

### Most Important Files:
1. 🔥 **`isaac_lab/envs/leatherback_env.py`** (739 lines) - Core RL environment
2. ⚙️ **`training/train_ppo.py`** (305 lines) - Main training script
3. 📊 **`isaac_lab/envs/leatherback_env_cfg.py`** (212 lines) - Environment config
4. 🔗 **`isaac_ros_bridge/isaac_publisher.py`** (227 lines) - Isaac → ROS2
5. 🎮 **`training/evaluate.py`** (244 lines) - Model evaluation

### Technology Stack:
- **Simulator**: Isaac Sim 5.1.0 + Isaac Lab
- **RL Framework**: Stable-Baselines3 (PPO)
- **Environment**: Gymnasium API
- **Robot**: NVIDIA Leatherback (Ackermann steering)
- **ROS2**: Jazzy (Ubuntu 24.04)
- **Python**: 3.11
- **GPU**: CUDA 12.8, PyTorch 2.7.0

### Training Capabilities:
- **Parallel Envs**: Up to 4096 on GPU (Isaac Lab)
- **Observation**: Dict space (vector + optional camera/LiDAR)
- **Action**: 2D continuous (throttle, steering)
- **Task**: Waypoint navigation (10 goals, 50m total)
- **Reward**: Progress + goal bonus - time penalty

---

## How to Use This Documentation

### For Training:
1. Read **Batch 1** (configs) for requirements
2. Follow **Batch 2** (setup) for installation
3. Understand **Batch 3** (environment) for RL mechanics
4. Use **Batch 4** (training) to train models

### For ROS2 Integration:
5. Add **Batch 5** (bridge) for ROS2 communication

### For Additional Features:
6. Reference **Batches 6-9** (utilities) as needed

---

## Next Steps

With this documentation, you can:
- ✅ Understand the entire project structure
- ✅ Set up the development environment
- ✅ Train RL agents with Isaac Sim
- ✅ Evaluate and visualize trained models
- ✅ Integrate with ROS2 (optional)
- ✅ Extend with custom features

**Happy Training! 🚀**

---
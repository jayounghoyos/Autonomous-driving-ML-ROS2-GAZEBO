#!/bin/bash
# =============================================================================
# Isaac Sim 5.1.0 (Standalone) + Isaac Lab + ROS2 Jazzy Environment Setup
# Ubuntu 24.04 LTS
# =============================================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo " Leatherback RL Environment Setup"
echo " Isaac Sim 5.1.0 (Standalone) + ROS2 Jazzy"
echo "=============================================="

# -----------------------------------------------------------------------------
# 1. Isaac Sim 5.1.0 Standalone Installation
# -----------------------------------------------------------------------------
export ISAAC_SIM_PATH=${ISAAC_SIM_PATH:-$HOME/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64}

if [ -d "$ISAAC_SIM_PATH" ]; then
    echo -e "${GREEN}[OK]${NC} Isaac Sim: $ISAAC_SIM_PATH"

    # Source Isaac Sim environment
    if [ -f "$ISAAC_SIM_PATH/setup_python_env.sh" ]; then
        source "$ISAAC_SIM_PATH/setup_python_env.sh"
        echo -e "${GREEN}[OK]${NC} Isaac Sim Python environment sourced"
    fi

    # Set Isaac Python alias
    export ISAAC_PYTHON="$ISAAC_SIM_PATH/python.sh"

    # Add Isaac Sim to path
    export PATH="$ISAAC_SIM_PATH:$PATH"

else
    echo -e "${RED}[ERROR]${NC} Isaac Sim not found at $ISAAC_SIM_PATH"
    echo "    Please set ISAAC_SIM_PATH environment variable to your installation"
    echo "    Example: export ISAAC_SIM_PATH=~/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64"
fi

# -----------------------------------------------------------------------------
# 2. Isaac Lab Path (cloned from GitHub)
# -----------------------------------------------------------------------------
export ISAACLAB_PATH=${ISAACLAB_PATH:-$HOME/IsaacLab}

if [ -d "$ISAACLAB_PATH" ]; then
    echo -e "${GREEN}[OK]${NC} Isaac Lab: $ISAACLAB_PATH"

    # Add Isaac Lab to PYTHONPATH
    export PYTHONPATH="$ISAACLAB_PATH/source/extensions:$PYTHONPATH"
else
    echo -e "${YELLOW}[WARN]${NC} Isaac Lab not found at $ISAACLAB_PATH"
    echo "    Clone it: git clone https://github.com/isaac-sim/IsaacLab.git ~/IsaacLab"
    echo "    Install:  cd ~/IsaacLab && ./isaaclab.sh --install sb3"
fi

# -----------------------------------------------------------------------------
# 3. ROS2 Jazzy (Ubuntu 24.04 native)
# -----------------------------------------------------------------------------
if [ -f /opt/ros/jazzy/setup.bash ]; then
    source /opt/ros/jazzy/setup.bash
    echo -e "${GREEN}[OK]${NC} ROS2 Jazzy sourced"
else
    echo -e "${YELLOW}[WARN]${NC} ROS2 Jazzy not found at /opt/ros/jazzy"
    echo "    Install: sudo apt install ros-jazzy-desktop"
fi

# -----------------------------------------------------------------------------
# 4. Project Setup
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo -e "${GREEN}[OK]${NC} Project: $PROJECT_ROOT"

# -----------------------------------------------------------------------------
# 5. ROS2 Workspace (if built)
# -----------------------------------------------------------------------------
if [ -f "$PROJECT_ROOT/ros2_ws/install/setup.bash" ]; then
    source "$PROJECT_ROOT/ros2_ws/install/setup.bash"
    echo -e "${GREEN}[OK]${NC} ROS2 workspace sourced"
fi

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
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
echo ""
echo "Or launch Isaac Sim directly:"
echo "  \$ISAAC_SIM_PATH/isaac-sim.sh"
echo ""

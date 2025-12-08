#!/bin/bash
# Isaac Sim Environment Setup
# Uses existing Isaac Sim 5.0.0 installation

# Isaac Sim installation path
export ISAAC_SIM_PATH=~/isaac-sim/isaac-sim-standalone-5.0.0-linux-x86_64

# Add Isaac Sim Python to path
export ISAAC_PYTHON=$ISAAC_SIM_PATH/python.sh

# Source ROS2 (Optional - Check path first)
# source /opt/ros/kilted/setup.bash

# Verify setup
echo "✓ Isaac Sim Path: $ISAAC_SIM_PATH"
# echo "✓ Isaac Python: $($ISAAC_PYTHON --version)" # Isaac Python can be noisy
echo "✓ SETUP COMPLETE"

# Ready to use
echo ""
echo "Environment ready! Use:"
echo "  \$ISAAC_PYTHON your_script.py"

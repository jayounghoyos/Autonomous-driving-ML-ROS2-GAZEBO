#!/bin/bash
# Create all ROS 2 packages for the autonomous driving project

set -e

cd ~/kaiju_ws

# Fix permissions
sudo chown -R $USER:$USER src datasets training

# Create package directories
mkdir -p src/vehicle_description/{urdf,meshes,config}
mkdir -p src/vehicle_gazebo/{worlds,models,launch,config}
mkdir -p src/ml_perception/{ml_perception,resource,config,models,launch}
mkdir -p src/data_collection/{data_collection,resource}
mkdir -p src/vehicle_control/{vehicle_control,resource}
mkdir -p src/autonomous_nav/{autonomous_nav,resource}

echo "✅ Package directories created!"
echo ""
echo "Next: Run './run.sh' and inside the container run 'colcon build'"

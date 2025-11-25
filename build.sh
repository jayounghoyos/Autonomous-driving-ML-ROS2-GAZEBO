#!/bin/bash
# Build the Docker image for ROS 2 Kilted + Gazebo Ionic + ML

set -e

echo "🐳 Building Kaiju Autonomous Driving Docker Image..."
echo "This may take 10-15 minutes on first build..."

# Allow X11 connections from Docker
xhost +local:docker > /dev/null 2>&1 || true

# Create X11 auth file for Docker
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - > /dev/null 2>&1 || true

# Build the image
docker compose build

echo ""
echo "✅ Build complete!"
echo ""
echo "Next steps:"
echo "  1. Run './run.sh' to start the container"
echo "  2. Inside container, build workspace: 'colcon build'"
echo "  3. Test Gazebo: 'gz sim shapes.sdf'"

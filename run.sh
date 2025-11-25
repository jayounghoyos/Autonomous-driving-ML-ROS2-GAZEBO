#!/bin/bash
# Run the Kaiju development container

set -e

# Allow X11 connections from Docker
xhost +local:docker > /dev/null 2>&1 || true

# Create X11 auth file for Docker
touch /tmp/.docker.xauth
xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f /tmp/.docker.xauth nmerge - > /dev/null 2>&1 || true

echo "🚀 Starting Kaiju Autonomous Driving Container..."
echo ""

# Check if container is already running
if [ "$(docker ps -q -f name=kaiju_workspace)" ]; then
    echo "Container is already running. Attaching..."
    docker exec -it kaiju_workspace /bin/bash
else
    echo "Starting new container..."
    docker compose up -d
    docker exec -it kaiju_workspace /bin/bash
fi

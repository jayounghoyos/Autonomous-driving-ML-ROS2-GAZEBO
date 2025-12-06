#!/bin/bash
echo "🧹 Killing all Gazebo and ROS processes..."
pkill -9 -f gz
pkill -9 -f ruby
pkill -9 -f ros
pkill -9 -f python3
echo " Cleanup complete. You can restart now."

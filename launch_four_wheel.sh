#!/bin/bash
# launch_four_wheel.sh - Launch 4-wheel robot with GUI

export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/kilted/opt/gz_sim_vendor/lib/gz-sim-9/plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH
export GZ_SIM_RESOURCE_PATH=$(pwd)/src/vehicle_description/urdf:$(pwd)/src/vehicle_gazebo/models

echo "Launching Gazebo WITH GUI..."
gz sim -r src/vehicle_gazebo/worlds/rl_training_world.sdf &

sleep 5

echo "Spawning 4-Wheel Robot..."
ros2 run ros_gz_sim create \
    -world rl_training_world \
    -file ackermann_rl_car.sdf \
    -name ackermann_rl_car \
    -x 0 -y 0 -z 0.18

sleep 2

echo "Starting Bridge (4 wheels + odometry + camera)..."
ros2 run ros_gz_bridge parameter_bridge \
    /cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist \
    /scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
    /model/goal_marker/pose@geometry_msgs/msg/Pose]gz.msgs.Pose \
    /odom@nav_msgs/msg/Odometry[gz.msgs.Odometry \
    /camera/image_raw@sensor_msgs/msg/Image[gz.msgs.Image &

echo "GUI simulation ready! Camera + Odometry active"
wait

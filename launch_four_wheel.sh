#!/bin/bash
# launch_four_wheel.sh - Launch 4-wheel robot with GUI

export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/kilted/opt/gz_sim_vendor/lib/gz-sim-9/plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH

echo "Launching Gazebo WITH GUI..."
gz sim -r src/vehicle_gazebo/worlds/rl_training_world.sdf &

sleep 5

echo "Spawning 4-Wheel Robot..."
ros2 run ros_gz_sim create \
    -world rl_training_world \
    -file four_wheel_robot.sdf \
    -name four_wheel_robot \
    -x 0 -y 0 -z 0.3

sleep 2

echo "Starting Bridge (4 wheels + odometry)..."
ros2 run ros_gz_bridge parameter_bridge \
    /model/four_wheel_robot/joint/front_left_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /model/four_wheel_robot/joint/rear_left_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /model/four_wheel_robot/joint/front_right_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /model/four_wheel_robot/joint/rear_right_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /odom@nav_msgs/msg/Odometry[gz.msgs.Odometry &

echo "GUI simulation ready! Run training script"
wait

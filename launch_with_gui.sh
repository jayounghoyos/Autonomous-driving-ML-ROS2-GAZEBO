#!/bin/bash
# launch_with_gui.sh - Launch with GUI for visualization

export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/kilted/opt/gz_sim_vendor/lib/gz-sim-9/plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH

echo " Launching Gazebo WITH GUI..."
gz sim -r src/vehicle_gazebo/worlds/rl_training_world.sdf &

sleep 5

echo " Spawning Simple Robot..."
ros2 run ros_gz_sim create \
    -world rl_training_world \
    -file simple_robot.sdf \
    -name simple_robot \
    -x 0 -y 0 -z 0.3

sleep 2

echo " Starting Bridge (no LiDAR)..."
ros2 run ros_gz_bridge parameter_bridge \
    /model/simple_robot/joint/left_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /model/simple_robot/joint/right_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /odom@nav_msgs/msg/Odometry[gz.msgs.Odometry &

echo " GUI simulation ready! Run: python3 goal_seeking_env.py"
wait

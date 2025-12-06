#!/bin/bash
# launch_headless.sh - Launch Gazebo in headless mode (no GUI, no crash!)

export GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/ros/kilted/opt/gz_sim_vendor/lib/gz-sim-9/plugins:$GZ_SIM_SYSTEM_PLUGIN_PATH

echo " Launching Gazebo HEADLESS (no GUI)..."
gz sim -s -r -v 0 src/vehicle_gazebo/worlds/rl_training_world.sdf &

sleep 3

echo " Spawning Simple Robot..."
ros2 run ros_gz_sim create \
    -world rl_training_world \
    -file simple_robot.sdf \
    -name simple_robot \
    -x 0 -y 0 -z 0.3

sleep 2

echo " Starting Bridge..."
ros2 run ros_gz_bridge parameter_bridge \
    /model/simple_robot/joint/left_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /model/simple_robot/joint/right_wheel_joint/cmd_force@std_msgs/msg/Float64]gz.msgs.Double \
    /lidar@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan \
    /odom@nav_msgs/msg/Odometry[gz.msgs.Odometry &

echo " Headless simulation ready! Run python test_rl_env.py"
wait

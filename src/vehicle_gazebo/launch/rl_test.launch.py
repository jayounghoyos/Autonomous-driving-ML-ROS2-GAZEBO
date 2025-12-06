#!/usr/bin/env python3
"""
Launch file for RL training test environment
Launches Gazebo + ROS 2 bridges
"""

from launch import LaunchDescription
from launch.actions import ExecuteProcess, DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # Get path to world file
    pkg_dir = get_package_share_directory('vehicle_gazebo')
    world_file = os.path.join(
        os.path.dirname(pkg_dir),
        'vehicle_gazebo',
        'worlds',
        'rl_test_world.sdf'
    )

    # If that doesn't work, try direct path
    if not os.path.exists(world_file):
        world_file = os.path.join(
            os.getcwd(),
            'src',
            'vehicle_gazebo',
            'worlds',
            'rl_test_world.sdf'
        )

    return LaunchDescription([
        # Launch Gazebo
        ExecuteProcess(
            cmd=['gz', 'sim', world_file],
            output='screen'
        ),

        # Bridge /scan (LiDAR)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan'],
            output='screen'
        ),

        # Bridge /odom
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/odom@nav_msgs/msg/Odometry[gz.msgs.Odometry'],
            output='screen'
        ),

        # Bridge /cmd_vel (bidirectional: ROS 2 ↔ Gazebo)
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            arguments=['/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist'],
            output='screen'
        ),
    ])

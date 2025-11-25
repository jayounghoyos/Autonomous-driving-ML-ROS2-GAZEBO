#!/usr/bin/env python3

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    # Get package directories
    pkg_vehicle_gazebo = get_package_share_directory('vehicle_gazebo')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    # Paths
    world_file = os.path.join(pkg_vehicle_gazebo, 'worlds', 'test_track.sdf')

    # Launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value=world_file,
        description='Path to world file'
    )

    # Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={
            'gz_args': [LaunchConfiguration('world'), ' -r']
        }.items()
    )

    # Spawn vehicle (Ackermann steering version)
    pkg_vehicle_description = get_package_share_directory('vehicle_description')
    vehicle_sdf = os.path.join(pkg_vehicle_description, 'urdf', 'ackermann_car.sdf')

    spawn_vehicle = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'ackermann_car',
            '-file', vehicle_sdf,
            '-x', '0',
            '-y', '0',
            '-z', '0.5'
        ],
        output='screen'
    )

    # ROS-Gazebo bridge for camera
    bridge_camera = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo'
        ],
        output='screen'
    )

    # ROS-Gazebo bridge for cmd_vel and odom
    bridge_control = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/cmd_vel@geometry_msgs/msg/Twist@gz.msgs.Twist',
            '/odom@nav_msgs/msg/Odometry@gz.msgs.Odometry'
        ],
        output='screen'
    )

    # ROS-Gazebo bridge for TF transforms
    bridge_tf = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/ackermann_car/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V'
        ],
        output='screen',
        remappings=[
            ('/model/ackermann_car/tf', '/tf')
        ]
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        spawn_vehicle,
        bridge_camera,
        bridge_control,
        bridge_tf
    ])

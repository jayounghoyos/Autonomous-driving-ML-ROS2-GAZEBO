from setuptools import find_packages, setup

package_name = 'autonomous_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'rclpy',
        'std_msgs',
        'geometry_msgs',
        'vision_msgs',
    ],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='Autonomous navigation and path planning',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'lane_follower = autonomous_nav.lane_follower:main',
            'waypoint_follower = autonomous_nav.waypoint_follower:main',
            'obstacle_avoider = autonomous_nav.obstacle_avoider:main',
        ],
    },
)

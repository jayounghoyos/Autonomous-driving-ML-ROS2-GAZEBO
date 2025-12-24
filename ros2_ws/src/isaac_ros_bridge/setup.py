"""Setup for isaac_ros_bridge package."""

from setuptools import find_packages, setup

package_name = "isaac_ros_bridge"

setup(
    name=package_name,
    version="2.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="jayounghoyos",
    maintainer_email="jayoungh@eafit.edu.co",
    description="Bridge between Isaac Sim 5.1.0 and ROS2 Jazzy",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "isaac_publisher = isaac_ros_bridge.isaac_publisher:main",
            "ros_subscriber = isaac_ros_bridge.ros_subscriber:main",
        ],
    },
)

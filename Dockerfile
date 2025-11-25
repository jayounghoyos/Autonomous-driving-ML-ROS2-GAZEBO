# ROS 2 Kilted Kaiju + Gazebo Ionic + ML Environment
# Base: Ubuntu 24.04 (Noble) - required for Kilted
FROM ubuntu:24.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=kilted

# Set locale
RUN apt-get update && apt-get install -y \
    locales \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8

# Install basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg2 \
    lsb-release \
    software-properties-common \
    git \
    vim \
    nano \
    build-essential \
    cmake \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Add ROS 2 repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 Kilted Kaiju
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-desktop \
    ros-${ROS_DISTRO}-ros-base \
    python3-colcon-common-extensions \
    python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# Install Gazebo Ionic (via ros_gz bridge)
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-ros-gz \
    && rm -rf /var/lib/apt/lists/*

# Install additional ROS 2 packages for autonomous driving
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-navigation2 \
    ros-${ROS_DISTRO}-nav2-bringup \
    ros-${ROS_DISTRO}-rviz2 \
    ros-${ROS_DISTRO}-robot-localization \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-image-transport \
    ros-${ROS_DISTRO}-vision-opencv \
    ros-${ROS_DISTRO}-sensor-msgs-py \
    && rm -rf /var/lib/apt/lists/*

# Install ML/AI dependencies with CUDA support
# Use --ignore-installed to avoid conflicts with system packages
RUN pip3 install --no-cache-dir --break-system-packages --ignore-installed \
    torch torchvision torchaudio \
    opencv-python-headless \
    ultralytics \
    onnx \
    onnxruntime-gpu

# Initialize rosdep
RUN rosdep init || true \
    && rosdep update

# Set up workspace directory
WORKDIR /workspace

# Source ROS 2 setup in bashrc
RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc \
    && echo "if [ -f /workspace/install/setup.bash ]; then source /workspace/install/setup.bash; fi" >> ~/.bashrc

# Set up environment variables for Gazebo
ENV GZ_VERSION=ionic
ENV IGN_GAZEBO_RESOURCE_PATH=/workspace/src/vehicle_gazebo/models

# Install sudo for convenience
RUN apt-get update \
    && apt-get install -y sudo \
    && rm -rf /var/lib/apt/lists/*

# For now, run as root (we'll add user management later if needed)
# This avoids permission issues and speeds up development

CMD ["/bin/bash"]

#!/bin/bash
# Install NVIDIA Container Toolkit for Pop!_OS 22.04

set -e

echo "🔧 Installing NVIDIA Container Toolkit for Pop!_OS 22.04..."
echo ""

# Remove any broken config
sudo rm -f /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Use Ubuntu 22.04 repository (Pop!_OS is based on Ubuntu 22.04)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/\$(ARCH) /" | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Update and install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

echo ""
echo "✅ NVIDIA Container Toolkit installed!"
echo ""
echo "Testing GPU in Docker..."
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

echo ""
echo "🎉 GPU setup complete! Now rebuild your Docker image with ./build.sh"

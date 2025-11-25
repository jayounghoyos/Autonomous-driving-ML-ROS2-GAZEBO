#!/bin/bash
# Check NVIDIA GPU and Docker GPU support

echo "🔍 Checking NVIDIA GPU Setup..."
echo ""

# Check if nvidia-smi exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Please install NVIDIA drivers:"
    echo "   sudo apt update"
    echo "   sudo apt install nvidia-driver-535"
    exit 1
fi

# Check GPU
echo "✅ NVIDIA Driver installed:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

echo "✅ Docker installed: $(docker --version)"
echo ""

# Check NVIDIA Container Toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Container Toolkit not installed or not configured."
    echo ""
    echo "To install:"
    echo "  1. Install NVIDIA Container Toolkit:"
    echo "     distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
    echo "     curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
    echo "     curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | \\"
    echo "       sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \\"
    echo "       sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
    echo "     sudo apt update"
    echo "     sudo apt install -y nvidia-container-toolkit"
    echo ""
    echo "  2. Configure Docker:"
    echo "     sudo nvidia-ctk runtime configure --runtime=docker"
    echo "     sudo systemctl restart docker"
    exit 1
fi

echo "✅ NVIDIA Container Toolkit working!"
echo ""
echo "🎉 All GPU requirements satisfied! You can now run:"
echo "   ./build.sh"

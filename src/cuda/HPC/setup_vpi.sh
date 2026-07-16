#!/bin/bash
# VPI Setup Script
# Installs VPI library if not already present
# Requires sudo for system-wide installation

set -e

echo "VPI Setup Script"
echo "================"

# Check if VPI is already installed
if command -v vpi-config &> /dev/null; then
    VPI_VERSION=$(vpi-config --version)
    VPI_PATH=$(vpi-config --prefix)
    echo "VPI already installed: version $VPI_VERSION at $VPI_PATH"
    exit 0
fi

# Check if running on x86_64 Linux
if [[ "$(uname -m)" != "x86_64" ]] || [[ "$(uname -s)" != "Linux" ]]; then
    echo "ERROR: VPI installation is only supported on Linux x86_64"
    exit 1
fi

# Detect Ubuntu version
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    UBUNTU_VERSION=$VERSION_ID
else
    echo "ERROR: Cannot detect Ubuntu version"
    exit 1
fi

echo "Detected Ubuntu $UBUNTU_VERSION"

# Check for sudo
if ! command -v sudo &> /dev/null; then
    echo "ERROR: sudo is required for VPI installation"
    exit 1
fi

echo ""
echo "Installing VPI via apt..."
echo "This requires sudo privileges and will install system packages."
echo ""

# Install prerequisites
sudo apt-get update
sudo apt-get install -y gnupg software-properties-common

# Add NVIDIA repository key (modern method)
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://repo.download.nvidia.com/jetson/jetson-ota-public.asc | sudo gpg --dearmor -o /etc/apt/keyrings/nvidia-jetson.gpg

# Add repository based on Ubuntu version
if [[ "$UBUNTU_VERSION" == "22.04" ]]; then
    echo "deb [signed-by=/etc/apt/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/jammy r38.4 main" | sudo tee /etc/apt/sources.list.d/nvidia-jetson.list
elif [[ "$UBUNTU_VERSION" == "24.04" ]]; then
    echo "deb [signed-by=/etc/apt/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/noble r38.4 main" | sudo tee /etc/apt/sources.list.d/nvidia-jetson.list
else
    echo "WARNING: Ubuntu $UBUNTU_VERSION not officially supported. Trying jammy repository..."
    echo "deb [signed-by=/etc/apt/keyrings/nvidia-jetson.gpg] https://repo.download.nvidia.com/jetson/x86_64/jammy r38.4 main" | sudo tee /etc/apt/sources.list.d/nvidia-jetson.list
fi

# Install VPI packages
sudo apt-get update
sudo apt-get install -y libnvvpi4 vpi4-dev vpi4-samples

# Detect Python version and install Python bindings
PYTHON_VERSION=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
if [[ "$PYTHON_VERSION" == "3.10" ]]; then
    sudo apt-get install -y python3.10-vpi4
elif [[ "$PYTHON_VERSION" == "3.12" ]]; then
    sudo apt-get install -y python3.12-vpi4
else
    echo "WARNING: Python VPI bindings not available for Python $PYTHON_VERSION"
    echo "Supported versions: 3.10, 3.12"
fi

echo ""
echo "VPI installation complete!"
echo "VPI installed at: $(vpi-config --prefix)"
echo "VPI version: $(vpi-config --version)"

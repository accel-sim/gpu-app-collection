#!/bin/bash
# Link VPI sample data from VPI installation

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="$SCRIPT_DIR/../../../data_dirs/cuda/H100"

# Find VPI installation
VPI_ROOT=$(find /opt/nvidia -maxdepth 1 -name "vpi*" -type d 2>/dev/null | head -1)

if [ -z "$VPI_ROOT" ]; then
    echo "ERROR: VPI installation not found in /opt/nvidia/"
    exit 1
fi

VPI_ASSETS="$VPI_ROOT/samples/assets"

if [ ! -d "$VPI_ASSETS" ]; then
    echo "ERROR: VPI sample assets not found at $VPI_ASSETS"
    exit 1
fi

echo "Linking VPI sample data from $VPI_ASSETS..."

# Create data directory structure
mkdir -p "$DATA_ROOT/vpi"

# Remove existing data symlink if it exists
rm -rf "$DATA_ROOT/vpi/data"

# Create symbolic link to VPI assets
ln -s "$VPI_ASSETS" "$DATA_ROOT/vpi/data"

echo "VPI data linked successfully at $DATA_ROOT/vpi/data -> $VPI_ASSETS"

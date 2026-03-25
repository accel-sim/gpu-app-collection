#!/bin/bash
# Find the latest cuGraph tag that supports the given CUDA version

CUDA_VERSION=$1
CUGRAPH_DIR=$2

if [ -z "$CUDA_VERSION" ] || [ -z "$CUGRAPH_DIR" ]; then
    echo "Usage: $0 <cuda_version> <cugraph_dir>"
    exit 1
fi

cd "$CUGRAPH_DIR" || exit 1

# Fetch all tags
git fetch --tags --quiet 2>/dev/null

# Get all tags sorted by version (newest first)
TAGS=$(git tag -l 'v*' | sort -V -r)

# For each tag, check if it supports the CUDA version
for TAG in $TAGS; do
    # Checkout the tag quietly
    git checkout "$TAG" --quiet 2>/dev/null || continue

    # Check rapids-cmake or CMakeLists.txt for CUDA version support
    # Look for CUDA version specifications in cmake files
    if [ -f "rapids-cmake/rapids-cuda/rapids_cuda_init_architectures.cmake" ]; then
        CUDA_FILE="rapids-cmake/rapids-cuda/rapids_cuda_init_architectures.cmake"
    elif [ -f "cpp/CMakeLists.txt" ]; then
        CUDA_FILE="cpp/CMakeLists.txt"
    else
        continue
    fi

    # Extract supported CUDA versions from the file
    # Look for patterns like "CUDA 12.8" or "CUDA_VERSION 12.8"
    if grep -q "$CUDA_VERSION" "$CUDA_FILE" 2>/dev/null || \
       grep -qE "CUDA.*$CUDA_VERSION|$CUDA_VERSION.*CUDA" "$CUDA_FILE" 2>/dev/null; then
        echo "$TAG"
        exit 0
    fi
done

# If no tag found, return the latest tag
LATEST=$(git tag -l 'v*' | sort -V -r | head -1)
echo "$LATEST"

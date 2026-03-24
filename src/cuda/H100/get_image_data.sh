#!/bin/bash
# Generate test images for recursiveGaussian

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../../data_dirs/cuda/H100/recursiveGaussian/data"
mkdir -p "$DATA_DIR"

# Generate test PPM images using Python (portable, no ImageMagick dependency)
python3 - "$DATA_DIR" << 'EOF'
import os
import sys

def create_gradient_ppm(filename, size):
    """Create a simple gradient PPM image"""
    with open(filename, 'w') as f:
        f.write(f"P3\n{size} {size}\n255\n")
        for y in range(size):
            for x in range(size):
                # Gradient from black to white
                val = int((x + y) * 255 / (2 * size))
                f.write(f"{val} {val} {val} ")
            f.write("\n")

data_dir = os.path.expanduser(sys.argv[1])
create_gradient_ppm(f"{data_dir}/teapot128.ppm", 128)
create_gradient_ppm(f"{data_dir}/teapot256.ppm", 256)
create_gradient_ppm(f"{data_dir}/teapot512.ppm", 512)
print(f"Generated test images in {data_dir}")
EOF

echo "Image data ready in $DATA_DIR"

# Generate large test images using the Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/generate_large_ppm.py" ]; then
    echo "Generating large PPM images..."
    python3 "$SCRIPT_DIR/generate_large_ppm.py"
fi

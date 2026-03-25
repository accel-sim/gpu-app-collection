#!/bin/bash
# Generate test images for recursiveGaussian

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../../data_dirs/cuda/HPC/recursiveGaussian/data/"
mkdir -p "$DATA_DIR"

# Generate test PPM images using Python (portable, no ImageMagick dependency)
python3 - "$DATA_DIR" << 'EOF'
import os
import sys

def create_gradient_ppm(filename, size):
    """Create a simple gradient PPM image in P6 (binary) format"""
    with open(filename, 'wb') as f:
        # Write header in ASCII
        header = f"P6\n{size} {size}\n255\n"
        f.write(header.encode('ascii'))
        # Write pixel data in binary
        for y in range(size):
            for x in range(size):
                # Gradient from black to white
                val = int((x + y) * 255 / (2 * size))
                # Write RGB as 3 bytes (RGBA would need 4th byte)
                f.write(bytes([val, val, val]))

data_dir = os.path.expanduser(sys.argv[1])
create_gradient_ppm(f"{data_dir}/teapot128.ppm", 128)
create_gradient_ppm(f"{data_dir}/teapot256.ppm", 256)
create_gradient_ppm(f"{data_dir}/teapot512.ppm", 512)
create_gradient_ppm(f"{data_dir}/teapot768.ppm", 768)
create_gradient_ppm(f"{data_dir}/teapot1024.ppm", 1024)
print(f"Generated test images in {data_dir}")
EOF

echo "Image data ready in $DATA_DIR"

# Generate large test images using the Python script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/generate_large_ppm.py" ]; then
    echo "Generating large PPM images..."
    python3 "$SCRIPT_DIR/generate_large_ppm.py"
fi

#!/usr/bin/env python3
"""
Generate large PPM image files for recursiveGaussian benchmark.
Creates synthetic test images at various resolutions.
"""

import sys
import random

def generate_ppm(width, height, filename):
    """Generate a PPM P6 (binary RGB) image file."""

    print(f"Generating {width}x{height} PPM image...")

    # Calculate sizes
    pixel_count = width * height
    rgb_data_size = pixel_count * 3

    with open(filename, 'wb') as f:
        # Write ASCII header
        header = f"P6\n# Generated test image for recursiveGaussian_hpc\n{width} {height}\n255\n"
        f.write(header.encode('ascii'))

        # Generate RGB data in chunks to avoid memory issues
        chunk_size = 1024 * 1024  # 1MB chunks
        bytes_written = 0

        print(f"Writing {rgb_data_size / (1024*1024):.1f} MB of RGB data...")

        while bytes_written < rgb_data_size:
            # Generate chunk of random RGB values
            remaining = rgb_data_size - bytes_written
            current_chunk_size = min(chunk_size, remaining)

            # Create gradient pattern (more interesting than pure random)
            chunk_data = bytearray()
            for i in range(current_chunk_size // 3):
                pixel_idx = (bytes_written // 3) + i
                row = pixel_idx // width
                col = pixel_idx % width

                # Create a gradient pattern
                r = (col * 255 // width) & 0xFF
                g = (row * 255 // height) & 0xFF
                b = ((row + col) * 255 // (width + height)) & 0xFF

                chunk_data.extend([r, g, b])

            f.write(chunk_data)
            bytes_written += len(chunk_data)

            # Progress indicator
            progress = (bytes_written / rgb_data_size) * 100
            if bytes_written % (10 * 1024 * 1024) < chunk_size:  # Every ~10MB
                print(f"  Progress: {progress:.1f}%")

    # Get file size
    import os
    file_size = os.path.getsize(filename)

    print(f"✓ Created {filename}")
    print(f"  Size: {file_size / (1024*1024):.1f} MB")
    print(f"  Dimensions: {width}x{height}")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom size from command line
        if len(sys.argv) != 4:
            print("Usage: generate_large_ppm.py <width> <height> <output_file>")
            print("   or: generate_large_ppm.py (generates standard sizes)")
            sys.exit(1)

        width = int(sys.argv[1])
        height = int(sys.argv[2])
        filename = sys.argv[3]
        generate_ppm(width, height, filename)
    else:
        # Generate standard test sizes
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "../../../data_dirs/cuda/HPC/recursiveGaussian/data")

        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        sizes = [
            (768, 768, f"{base_dir}/teapot768.ppm"),
            (1024, 1024, f"{base_dir}/teapot1024.ppm"),
            # (2048, 2048, f"{base_dir}/teapot2048.ppm"),  # Uncomment for 4K
        ]

        for width, height, filename in sizes:
            generate_ppm(width, height, filename)

        print("All images generated successfully!")

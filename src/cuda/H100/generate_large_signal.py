#!/usr/bin/env python3
"""
Generate large signal files for dwtHaar1D benchmark.
Creates synthetic signal data at various sizes (powers of 2).
"""

import sys
import math

def generate_signal(size_power, filename, epsilon=0.001):
    """
    Generate a signal file with 2^size_power elements.

    Args:
        size_power: Power of 2 (e.g., 24 for 2^24 = 16,777,216 elements)
        filename: Output filename
        epsilon: Epsilon value for header (precision parameter)
    """

    num_elements = 2 ** size_power

    print(f"Generating signal file with {num_elements:,} elements (2^{size_power})...")
    print("This will take a few minutes...")

    with open(filename, 'w') as f:
        # Write header (epsilon value as comment)
        f.write(f"# {epsilon}\n")

        # Generate and write signal values
        # Using a simple synthetic signal (sine wave + noise)
        chunk_size = 100000  # Write in chunks
        values_written = 0

        while values_written < num_elements:
            chunk_values = []

            for i in range(min(chunk_size, num_elements - values_written)):
                idx = values_written + i

                # Create a synthetic signal: combination of multiple frequencies
                # This creates a more realistic signal than pure random
                t = idx / num_elements  # Normalized time 0 to 1

                # Multiple frequency components
                value = (
                    math.sin(2 * math.pi * 5 * t) * 0.5 +      # 5 Hz
                    math.sin(2 * math.pi * 13 * t) * 0.3 +     # 13 Hz
                    math.sin(2 * math.pi * 31 * t) * 0.2       # 31 Hz
                )

                chunk_values.append(f"{value:.6f}")

            # Write chunk (space-separated)
            f.write(" ".join(chunk_values))
            f.write(" ")

            values_written += len(chunk_values)

            # Progress indicator
            progress = (values_written / num_elements) * 100
            if progress % 25 < (100 * chunk_size / num_elements):
                print(f"Progress: {progress:.1f}%")

        f.write("\n")

    # Get file size
    import os
    file_size = os.path.getsize(filename)

    print(f"\n✓ Generated {filename}")
    print(f"  Elements: {num_elements:,} (2^{size_power})")
    print(f"  File size: {file_size / (1024*1024):.1f} MB")
    print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Custom size from command line
        if len(sys.argv) != 3:
            print("Usage: generate_large_signal.py <power_of_2> <output_file>")
            print("   Example: generate_large_signal.py 24 signal_2_24.dat")
            print("   or: generate_large_signal.py (generates standard sizes)")
            sys.exit(1)

        size_power = int(sys.argv[1])
        filename = sys.argv[2]
        generate_signal(size_power, filename)
    else:
        # Generate standard test sizes
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(script_dir, "../../../data_dirs/cuda/H100/dwtHaar1D/data")

        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)

        sizes = [
            (20, f"{base_dir}/signal_2_20.dat"),  # 2^20 = 1,048,576
            (22, f"{base_dir}/signal_2_22.dat"),  # 2^22 = 4,194,304
            (24, f"{base_dir}/signal_2_24.dat"),  # 2^24 = 16,777,216
        ]

        for power, filename in sizes:
            generate_signal(power, filename)

        print("All signal files generated successfully!")

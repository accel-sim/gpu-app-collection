#!/bin/bash
# Generate random signal data for DWT

DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../../data_dirs/cuda/H100/dwtHaar1D"
mkdir -p "$DATA_DIR"

# Generate random signal files (using dd for portability - no numpy required)
for size in 512 1024 4096 16384 65536; do
    dd if=/dev/urandom of="$DATA_DIR/signal_${size}.dat" bs=4 count=$size status=none 2>/dev/null
    echo "Generated signal_${size}.dat"
done

echo "DWT signal data ready"

echo "DWT data ready in $DATA_DIR"

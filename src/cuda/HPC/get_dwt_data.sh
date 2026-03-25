#!/bin/bash
# Generate random signal data for DWT

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data_dirs/cuda/HPC/dwtHaar1D/data/"
mkdir -p "$DATA_DIR"

# Link gold files and signal files from cuda-samples
CUDA_SAMPLES_DWT="$SCRIPT_DIR/../cuda-samples/Samples/5_Domain_Specific/dwtHaar1D/"

if [ -d "$CUDA_SAMPLES_DWT" ]; then
    echo "Linking reference data from cuda-samples..."
    # Link gold files (rename to match yml expectations)
    ln -sf "$CUDA_SAMPLES_DWT/data/regression_2_18.gold.dat" "$DATA_DIR/regression_2_18.gold.dat"
    ln -sf "$CUDA_SAMPLES_DWT/data/regression_2_14.gold.dat" "$DATA_DIR/regression_2_14.gold.dat"
    ln -sf "$CUDA_SAMPLES_DWT/data/regression.gold.dat" "$DATA_DIR/regression.gold.dat"

    # Link signal files from cuda-samples
    ln -sf "$CUDA_SAMPLES_DWT/data/signal_2_18.dat" "$DATA_DIR/signal_2_18.dat"
    ln -sf "$CUDA_SAMPLES_DWT/data/signal_2_14.dat" "$DATA_DIR/signal_2_14.dat"
    ln -sf "$CUDA_SAMPLES_DWT/data/signal.dat" "$DATA_DIR/signal.dat"

    echo "Linked reference data from cuda-samples"
fi

# Generate random signal files (using dd for portability - no numpy required)
for size in 512 1024 4096 16384 65536; do
    dd if=/dev/urandom of="$DATA_DIR/signal_${size}.dat" bs=4 count=$size status=none 2>/dev/null
    echo "Generated signal_${size}.dat"
done

echo "DWT signal data ready"

echo "DWT data ready in $DATA_DIR"

# Generate large signal files using the Python script
if [ -f "$SCRIPT_DIR/generate_large_signal.py" ]; then
    echo "Generating large signal files..."
    python3 "$SCRIPT_DIR/generate_large_signal.py"
fi

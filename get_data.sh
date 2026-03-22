#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_SUBDIR="/data_dirs/"
DATA_ROOT=$BASH_ROOT$DATA_SUBDIR

if [ ! -d $DATA_ROOT ]; then
	if [ ! -f $BASH_ROOT/all.gpgpu-sim-app-data.tgz ]; then
		wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/all.gpgpu-sim-app-data.tgz
	fi
    tar xzvf all.gpgpu-sim-app-data.tgz -C $BASH_ROOT
    rm all.gpgpu-sim-app-data.tgz
fi

# Generate H100 benchmark data
echo "Generating H100 benchmark data..."
if [ -f $BASH_ROOT/src/cuda/H100/get_graph_data.sh ]; then
    bash $BASH_ROOT/src/cuda/H100/get_graph_data.sh || echo "Warning: Graph data generation failed"
fi
if [ -f $BASH_ROOT/src/cuda/H100/get_image_data.sh ]; then
    bash $BASH_ROOT/src/cuda/H100/get_image_data.sh || echo "Warning: Image data generation failed"
fi
if [ -f $BASH_ROOT/src/cuda/H100/get_dwt_data.sh ]; then
    bash $BASH_ROOT/src/cuda/H100/get_dwt_data.sh || echo "Warning: DWT data generation failed"
fi

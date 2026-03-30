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

echo "Generating HPC benchmark data..."
if [ -f $BASH_ROOT/src/cuda/HPC/get_graph_data.sh ]; then
    bash $BASH_ROOT/src/cuda/HPC/get_graph_data.sh || echo "Warning: Graph data generation failed"
fi
if [ -f $BASH_ROOT/src/cuda/HPC/get_image_data.sh ]; then
    bash $BASH_ROOT/src/cuda/HPC/get_image_data.sh || echo "Warning: Image data generation failed"
fi
if [ -f $BASH_ROOT/src/cuda/HPC/get_dwt_data.sh ]; then
    bash $BASH_ROOT/src/cuda/HPC/get_dwt_data.sh || echo "Warning: DWT data generation failed"
fi
if [ -f $BASH_ROOT/src/cuda/HPC/get_vpi_data.sh ]; then
    bash $BASH_ROOT/src/cuda/HPC/get_vpi_data.sh
fi

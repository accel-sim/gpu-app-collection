#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_SUBDIR="/data_dirs/"
DATA_ROOT=$BASH_ROOT$DATA_SUBDIR

if [ ! -d $DATA_ROOT ]; then
	if [ ! -f $BASH_ROOT/all.gpgpu-sim-app-data.tgz ]; then
		wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/all.gpgpu-sim-app-data.tgz
	fi
    tar xzvf all.gpgpu-sim-app-data.tgz -C $BASH_ROOT
    tar xvzf dct.tgz -C $DATA_ROOT
fi

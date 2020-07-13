#!/bin/bash
export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_SUBDIR="/data_dirs/"
DATA_ROOT=$BASH_ROOT$DATA_SUBDIR

if [ ! -d $DATA_ROOT ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/rodinia_2.0-ft.data.tgz
    tar xzvf rodinia_2.0-ft.data.tgz -C $BASH_ROOT
    rm rodinia_2.0-ft.data.tgz
fi

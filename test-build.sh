#!/bin/bash

set -e

if [ ! -n "$CUDA_INSTALL_PATH" ]; then
»   echo "ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.";
»   exit;
fi

#Make the simulator
export PATH=$CUDA_INSTALL_PATH/bin:$PATH;

source src/setup_environment
make -j -C src/

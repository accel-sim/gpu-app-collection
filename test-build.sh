#!/bin/bash

set -e

if [ ! -n "$CUDA_INSTALL_PATH" ]; then
    echo "ERROR ** Install CUDA Toolkit and set CUDA_INSTALL_PATH.";
    exit 1;
fi

if [ ! -n "$BOOST_ROOT" ]; then
    echo "ERROR ** Install BOOST and set BOOST_ROOT.";
    exit 1;
fi

export PATH=$CUDA_INSTALL_PATH/bin:$PATH;

source src/setup_environment

MAKE_TARGET="${1:-all}"

make -C src/ "$MAKE_TARGET"

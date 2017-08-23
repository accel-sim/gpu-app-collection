#!/bin/bash
source /home/ascudiero/scratch/research/einstein/software/o-xterm.bash

export CUDA_USE_HOST_FE=1
export CUDA_ECX_ARGS="--stats 2"

/home/ascudiero/XSBench-ECX/XSBench 12 small



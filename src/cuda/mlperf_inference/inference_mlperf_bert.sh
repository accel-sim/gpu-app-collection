#!/bin/sh

ORIGINAL_FOLDER=$PWD
CUDA_VERSION=`nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/'`;
BASE_MLPERF_DIR=$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/mlperf_inference/
cd $BASE_MLPERF_DIR
. ./mlc/bin/activate
export MLC_SCRIPT_EXTRA_CMD="--adr.python.name=mlperf" &&
mlcr run-mlperf,inference,_full,_r5.0-dev \
   --model=bert-99 \
   --implementation=reference \
   --framework=pytorch \
   --category=edge \
   --scenario=Offline \
   --execution_mode=valid \
   --device=cuda \
   --quiet --rerun
deactivate
cd $ORIGINAL_FOLDER

# This is an executable script for tracing 
# a full run (all inference quires) for 
# BERT language model
# see https://docs.mlcommons.org/inference/benchmarks/language/bert/
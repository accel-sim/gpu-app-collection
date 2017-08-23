#!/bin/bash

if [ ! -z "$1" ]
then
	export ECX_MDF=$1
else
	export ECX_MDF="${EINSTEIN_BUILD_DIR}/ecx/mdfs/gryphon-1tpc.mdf"
fi
if [ ! -z "$2" ]
then
	export EMC_MM=$2
else
	export EMC_MM=''
fi

export CUDA_USE_HOST_FE=1
export CUDA_ECX_ARGS="--dump-knobs 1 --mdf $ECX_MDF ${CUDA_ECX_ARGS} "
export CUDA_IGNORE_MULTIPLE_CTX_INIT=1

echo "CUDA_ECX_ARGS ${CUDA_ECX_ARGS}"

./matvec inputs/64x64x64.mtx 0

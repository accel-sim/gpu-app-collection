#!/bin/bash

#default arguments
SIZE="small"
KERNEL=0
ECX_MDF="$EINSTEIN_ROOT/build_debug_64/ecx/mdfs/gryphon-1sm.mdf"

while getopts ":s:k:m:-:" arg; do
  case $arg in
    # size parameter for kernel 
    s)
      SIZE=${OPTARG}
      echo "setting size $SIZE"
      ;;
    # select kernel implementation
    k)
      KERNEL=${OPTARG}
      echo "setting kernel $KERNEL"
      ;;
    # set ECX MDF
    m)
      ECX_MDF=${OPTARG}  
      echo "setting ECX_MDF $ECX_MDF"
      ;;
    -)
      echo "WARNING: invalid option with '--' supplied (${OPTARG}) "
      echo "  IGNORING, not processing ANY further arguments, proceeding..."
      break;
      ;;
    *)
      echo "invalid option supplied, IGNORING, not processing any further arguments, proceeding..."
      break;
      ;;
  esac
done

export CUDA_USE_HOST_FE=1
export CUDA_ECX_ARGS="--stats 2 --mdf ${ECX_MDF} ${CUDA_ECX_ARGS} "
echo "CUDA_ECX_ARGS: ${CUDA_ECX_ARGS}"

echo "SIZE: $SIZE KERNEL:$KERNEL"
./XSBench 12 $SIZE $KERNEL > result.info

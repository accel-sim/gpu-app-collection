#!/bin/bash

#
#default arguments
#

SIZE="small"  #Problem size
KERNEL=0      #Kernel to run 
ECX_MDF="$EINSTEIN_ROOT/build_debug_64/ecx/mdfs/gryphon-1sm.mdf"   #default mdf

nx=64
ny=32
nz=32


usage()
{
cat << EOF
usage: $0 options

This script runs a specific CNS kernel with specific problem size.

OPTIONS:
   -h                        Show this message
   -k <kernel name>          compile a specific kernel 
   -s <small|medium|large>   problem size 
   -m <mdf file>             MDF file to use
EOF
}

echo "Starting $0 ...." 
echo "Script called with: \"$0 $@\""
echo ""
echo ""

while getopts "hs:k:m:-:" arg; do
  case $arg in
    h)
      usage 
      exit 1
      ;;
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

if [ $SIZE = "small" ]
then 
    nx=64
    ny=32
    nz=32
else
    echo "ERROR: Unrecognized size=$SIZE"
    exit 0 
fi

if [ $KERNEL = "0" ]
then 
    echo "ERROR: kernel was not defined"
    exit 0 
fi

export CUDA_USE_HOST_FE=1
export CUDA_ECX_ARGS="--stats 2 --mdf ${ECX_MDF} ${CUDA_ECX_ARGS} "
echo "CUDA_ECX_ARGS: ${CUDA_ECX_ARGS}"

echo "./$KERNEL $nx $ny $nz 1"
./$KERNEL $nx $ny $nz 1

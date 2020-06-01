#!/bin/bash

#List of kernels
KERNELS=( TRACE_DX TRACE_DZ TRACE_DIFFLUX_UX TRACE_DIFFLUX_UY TRACE_DIFFLUX_UZ TRACE_DIFFLUX_E TRACE_UPDATE_RHO TRACE_UPDATE_E TRACE_UPDATE_U )
#Flag for nuking all temp files
CLEAN=0 



usage()
{
cat << EOF
usage: $0 options

This script compiles CNS. It can either compile all kernels or one specific kernel.

OPTIONS:
   -h                  Show this message
   -c                  clean all temps and executables
   -k <kernel name>    compile a specific kernel 
EOF
}



while getopts "hk:c" arg; do
    case $arg in
    h)
      usage 
      exit 1
      ;;

    k)
      echo "single kernel" 
      KERNELS=(${OPTARG})
      ;; 
    c)
      echo "cleaning temp files"
      CLEAN=1
      ;;
   -)
      echo "WARNING: invalid option with '--' supplied (${OPTARG}) "
      break;
      ;;
    *)
      echo "WARNING: invalid option supplied, IGNORING, not processing any further arguments, proceeding..."
      ;;
    esac
done

#for i in ${KERNELS[@]}
#do
#  echo $i
#  if [ $CLEAN = 1 ]
#  then
#      rm -f $i
#      rm -f *_tmp.ptx *_tmp.s *_tmp.vir *_tmp.so *_tmp.ccbak *_tmp.os
#      
#  else
#      nvcc -Xptxas -v,-abi=no prototype_simple.cu -D$i -o $i
#      #nvcc -arch=sm_35 -Xptxas -v,-abi=no prototype.cu -D$i -o $i
#  fi
#done

for i in ${KERNELS[@]}
do
    FLAGS="$FLAGS -D$i"
done

#nvcc -arch=sm_20 -Xptxas -v,-abi=no prototype_simple.cu $FLAGS -o cns_all -lcudart
nvcc -Xptxas -v prototype_simple.cu $FLAGS -o cns_all -lcudart

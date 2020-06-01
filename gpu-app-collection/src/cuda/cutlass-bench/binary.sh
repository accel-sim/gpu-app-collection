#!/bin/bash
nvcc -O3 -DNDEBUG CMakeFiles/cutlass_perf_test.dir/cutlass_perf_test_generated_cutlass_perf_test.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_sgemm.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_sgemm_splitK.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_dgemm.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_hgemm.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_igemm.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_igemm_splitK.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_wmma_gemm.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_volta884_gemm.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_volta884_gemm_splitK.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_volta884_gemm_cta_rasterization_tn.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_volta884_gemm_cta_rasterization_tt.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_volta884_gemm_cta_rasterization_nn.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_volta884_gemm_cta_rasterization_nt.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_wmma_binary_gemm.cu.o CMakeFiles/cutlass_perf_test.dir/gemm/cutlass_perf_test_generated_wmma_integer_gemm.cu.o  -o cutlass_perf_test -lcudart

BIN_FOLDER=$(pwd)
for I in $(seq 1 284)
do
    if [[ ! -d ${I}  ]]; then
        mkdir $I
    fi
    cd ${I}
    ln -sf ../cutlass_perf_test .
    ln -sf /home/scratch.ziy_nvresearch/gpgpusim/gpgpusim-gcc-4.8.5-cuda-9.1.85/configs/tested-cfgs/SM7_QV100/* .
    cd ${BIN_FOLDER}
done
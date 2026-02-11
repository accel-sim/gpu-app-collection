#!/bin/bash
# A script to sweep the number of concurrent accesses to the L2 cache to test for the LRC max merged

export CUDA_VISIBLE_DEVICES=7
for tb_size in 32 64 128; do 
    echo "--------------------------------"
    echo "Thread block size: $tb_size"
    echo "NUM_CONCURRENT_ACCESS,NUM_BLOCKS,lts2lrc_sectors,xbar2gpc_sectors,LRC ratio(xbar2gpc/lts2lrc)"
    for nb in 1 4 8 16 32 48 64 80 96 112 128; do
        num_concurrent_access=$((nb * tb_size / 32))
        output=$(TMPDIR=./ ncu --metrics lrc__xbar2gpc_sectors_op_read.sum,lrc__lts2lrc_sectors_op_read.sum,lrc__average_xbar2gpc_sectors_op_read.ratio,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./lrc_max_merged "$nb" 128 2048 0 2>&1)

        lts2lrc=$(echo "$output" | grep 'lrc__lts2lrc_sectors_op_read.sum' | awk '{print $NF}' | tr -d ',')
        xbar2gpc=$(echo "$output" | grep 'lrc__xbar2gpc_sectors_op_read.sum' | awk '{print $NF}' | tr -d ',')

        if [[ -n "$lts2lrc" && -n "$xbar2gpc" && "$xbar2gpc" != "0" ]]; then
            ratio=$(awk "BEGIN {printf \"%.4f\", $xbar2gpc / $lts2lrc}")
        else
            ratio="N/A"
        fi

        echo "${num_concurrent_access},${nb},${lts2lrc},${xbar2gpc},${ratio}"
    done
    echo "--------------------------------"
    echo ""
done
#!/bin/bash
export CUDA_VISIBLE_DEVICES=7

# Measure maximum merge count per LRC entry
# Run with cluster launch with 8 threadblocks and 8 threadblocks per cluster
# Should see 8*128/32 = 32 sectors request issued from SM
# and 8 sectors request issued from L2 cache
TMPDIR=./ ncu \
	--metrics l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum \
	--replay-mode application ./lrc_max_merged -N 8 -C 8 -T 128 -m 1

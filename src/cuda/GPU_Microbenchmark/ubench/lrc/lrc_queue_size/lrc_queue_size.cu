// LRC Queue Size Microbenchmark — POSTPONED
//
// TODO: Implement benchmark to discover LRC queue depth (max_entries) per
// L2 sub-partition on NVIDIA GPUs.
//
// Challenges that need to be resolved before implementation:
//
// 1. Address-to-sub-partition mapping:
//    To measure the queue depth of a specific L2 sub-partition, we need to
//    generate unique sector addresses that all map to the SAME sub-partition.
//    This requires knowledge of the H100's L2 address interleaving scheme.
//    A simple linear mapping (addr / sector_size % num_sub_partitions) is
//    likely incorrect — NVIDIA may use XOR-based hashing or other schemes.
//
// 2. Measurement approach:
//    Once the address mapping is known, launch multiple warps that each read
//    from a unique sector address (all mapping to the same sub-partition).
//    Sweep the number of unique sectors and use ncu to observe:
//    - Pre-LRC: l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
//    - Post-LRC: lts__t_sectors_srcunit_tex_op_read.sum
//    When the number of unique sectors exceeds the LRC queue depth,
//    the LRC can no longer track all entries simultaneously, causing
//    evictions and additional L2 lookups.
//
// 3. Alternative approach:
//    Reverse-engineer the address interleaving first using a separate
//    microbenchmark (e.g., by measuring latency variation across different
//    address strides to identify the interleaving pattern).
//
// Accel-Sim config parameter: -gpgpu_lrc_max_entries <value>

#include <stdio.h>

int main() {
  printf("lrc_queue_size: NOT YET IMPLEMENTED\n");
  printf("See source file for TODO details.\n");
  return 0;
}

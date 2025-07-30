

#ifndef BLACK_B200_DEF_H
#define BLACK_B200_DEF_H

#include "./common/common.h"
#include "./common/deviceQuery.h"

#define L1_SIZE (256 * 1024) // Max L1 size in bytes

#define CLK_FREQUENCY 1665 // frequency in MHz

#define ISSUE_MODEL issue_model::single // single issue core or dual issue
#define CORE_MODEL core_model::subcore  // subcore model or shared model
#define DRAM_MODEL dram_model::HBM      // memory type
#define WARP_SCHEDS_PER_SM 4            // number of warp schedulers per SM


#define SASS_hmma_per_PTX_wmma 2


#define L2_BANKS_PER_MEM_CHANNEL 1
#define L2_BANK_WIDTH_in_BYTE 64

#endif

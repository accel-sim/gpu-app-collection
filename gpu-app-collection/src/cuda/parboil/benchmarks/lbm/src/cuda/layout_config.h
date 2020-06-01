/***************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*############################################################################*/

#ifndef _LAYOUT_CONFIG_H_
#define _LAYOUT_CONFIG_H_

/*############################################################################*/

//Unchangeable settings: volume simulation size for the given example
#define SIZE_X (120)
#define SIZE_Y (120)
#define SIZE_Z (150)

//Changeable settings
//Padding in each dimension
//Align rows of X to 128-bytes
#define PADDING_X (8)
#define PADDING_Y (0)
#define PADDING_Z (4)

//Pitch in each dimension
#define PADDED_X (SIZE_X+PADDING_X)
#define PADDED_Y (SIZE_Y+PADDING_Y)
#define PADDED_Z (SIZE_Z+PADDING_Z)

#define TOTAL_CELLS (SIZE_X*SIZE_Y*(SIZE_Z))
#define TOTAL_PADDED_CELLS (PADDED_X*PADDED_Y*(PADDED_Z))

//Flattening function
//  This macro will be used to map a 3-D index and element to a value
#define CALC_INDEX(x,y,z,e) ( TOTAL_PADDED_CELLS*e + \
                               ((x)+(y)*PADDED_X+(z)*PADDED_X*PADDED_Y) )

// Set this value to 1 for GATHER, or 0 for SCATTER
#if 0
#define GATHER
#else
#define SCATTER
#endif

//CUDA block size (not trivially changeable here)
#define BLOCK_SIZE SIZE_X

/*############################################################################*/

#endif /* _CONFIG_H_ */

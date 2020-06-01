#pragma once

#define BFS_LS 0
#define BFS_ATOMIC 1
#define BFS_MERRILL 2
#define BFS_WORKLISTW 3 // worklist version from worklist directory
#define BFS_WORKLISTG 4 // deleted experimental version
#define BFS_WORKLISTA 5  // bitmasks
#define BFS_WORKLISTC 6  // cub

#ifndef VARIANT
#error "VARIANT not defined."
#endif

#if VARIANT==BFS_LS
#include "bfs_ls.h"
#elif VARIANT==BFS_ATOMIC
#include "bfs_topo_atomic.h"
#elif VARIANT==BFS_MERRILL
#include "bfs_merrill.h"
#elif VARIANT==BFS_WORKLISTW
#include "bfs_worklistw.h"
#elif VARIANT==BFS_WORKLISTG
#include "bfs_worklistg.h"
#elif VARIANT==BFS_WORKLISTA
#include "bfs_worklista.h"
#elif VARIANT==BFS_WORKLISTC
#include "bfs_worklistc.h"
#else 
#error "Unknown variant"
#endif

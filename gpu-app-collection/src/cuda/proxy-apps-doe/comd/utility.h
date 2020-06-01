#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>

#include <cuda_runtime.h>

#include "interface.h"
#include "cmd_parser.h"
#include "lj_reader.h"
#include "eam_reader.h"
#include "sim_reader.h"
#include "cheby.h"
#include "timer.h"

// parse command line options
simflat_t *parse(int argc, char **argv, config &c);

// data manipulation on host
void create_sim_host(const config& c, simflat_t *simflat, sim_t *sim);
void dump_sim_info(sim_t &sim);

// data manipulation on device
void create_sim_dev(sim_t &sim_H, sim_t *sim_D);
void fill_sim_dev(sim_t &sim_H, sim_t &sim_D);
void destroy_sim_dev(sim_t &sim);

// compute system energy
void compute_energy(simflat_t *sim, sim_t &sim_D);


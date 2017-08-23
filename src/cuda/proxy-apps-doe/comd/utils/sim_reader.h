
#ifndef __READ_H
#define __READ_H

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>

#include "types.h"
#include "domains.h"

#define DEBUGLEVEL 0
#define PMDDEBUGPRINTF(xxx,...) {if(xxx>DEBUGLEVEL) printf(__VA_ARGS__);}

simflat_t *create_fcc_lattice(command_t cmd, struct pmd_base_potential_t *pot);

simflat_t *fromFileASCII(command_t cmd, struct pmd_base_potential_t *pot);
simflat_t *fromFileGzip(command_t cmd, struct pmd_base_potential_t *pot);
simflat_t *fromFileTim(command_t cmd, struct pmd_base_potential_t *pot);

#endif

/// \file
/// Compute forces for the Embedded Atom Model (EAM).

#ifndef __EAM_H
#define __EAM_H

#include "mytype.h"

struct BasePotentialSt;
struct LinkCellSt;

struct BasePotentialSt* initEamPot(const char* dir, const char* file, const char* type);
#endif

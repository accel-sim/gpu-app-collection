#ifndef __CHEBY_H_
#define __CHEBY_H_

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "types.h"

#define DIAG_CHEBY 0
#define PI 3.141592653589793

struct eam_cheby_t *setChebPot(eampotential_t *pot, int n);

struct potentialarray_t *genCheb(potentialarray_t *pot, int n);

struct potentialarray_t *genDer(potentialarray_t *ch);

real_t eamCheb(potentialarray_t *cheb, real_t x);

void chder(real_t a, real_t b, real_t *c, real_t *cder, int n);

void readArraySizes( eam_cheby_t *ch_pot, int* numCh, int* numRef);

void readMishinRef(
	int* numRef,
	real_t* phiRange,
	real_t* rhoRange,
	real_t* FRange,
	real_t* phiRef, 
	real_t* rhoRef, 
	real_t* FRef, 
	real_t* xphiRef, 
	real_t* xrhoRef, 
	real_t* xFRef) ;

void readMishinCh(eam_cheby_t *ch_pot);

void genDerivs(eam_cheby_t *ch_pot);

void computeCompare(
	real_t* range, 
	real_t* ch, 
	real_t* dch,
	real_t* ref, 
	real_t* x, 
	real_t* ref_val_h,
	real_t* ref_dval_h,
	int n_ref, 
	int n_ch);

static void eamInterpolateDerivlocal(real_t r,
	real_t* values,
	int n_values,
	real_t *range,
	real_t *value1, 
	real_t *f1);

#endif

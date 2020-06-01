
#ifndef __EAM_H
#define __EAM_H

#include <math.h>

#include "types.h"

eampotential_t *eamReadASCII(char *dir, char *potname);
void eamDestroy(void **inppot);
int eamForce(void *s);

static void destroyPotentialArray(struct potentialarray_t **a, int doubleFlag);

struct pmd_base_potential_t *setEamPot(char *dir, char *file);

struct eampotential_t *getEamPot();

static struct potentialarray_t *allocPotentialArray(int n, real_t x0, real_t xn, real_t invDx);

static struct potentialarray_t *getPotentialArrayFromBinaryFile(char *file);

static struct potentialarray_t *getPotentialArrayFromFile(char *file);

static double getMassFromFile(char *file);

static double getLatFromFile(char *file);

eampotential_t *eamReadASCII(char *dir, char *potname);

real_t eamCheb(potentialarray_t *cheb, real_t x);

static inline void eamInterpolateDeriv(struct potentialarray_t *a, real_t r, int iType, int jType, real_t *value1, real_t *f1);

static void adiffpot(char *name,potentialarray_t *a, potentialarray_t *b);

void eamComparePots(eampotential_t *a, eampotential_t *b);


#endif

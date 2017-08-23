/*

Copyright (c) 2011, Los Alamos National Security, LLC All rights
reserved.  Copyright 2011. Los Alamos National Security, LLC. This
software was produced under U.S. Government contract DE-AC52-06NA25396
for Los Alamos National Laboratory (LANL), which is operated by Los
Alamos National Security, LLC for the U.S. Department of Energy. The
U.S. Government has rights to use, reproduce, and distribute this
software.

NEITHER THE GOVERNMENT NOR LOS ALAMOS NATIONAL SECURITY, LLC MAKES ANY
WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF
THIS SOFTWARE.

If software is modified to produce derivative works, such modified
software should be clearly marked, so as not to confuse it with the
version available from LANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

· Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.

· Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.

· Neither the name of Los Alamos National Security, LLC, Los Alamos
  National Laboratory, LANL, the U.S. Government, nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING,
BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS
ALAMOS NATIONAL SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

/**
 * An interface for reading afv style potential files
 *
 * Written by Sriram Swaminarayan 9/11/2006
 **/

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "eam_reader.h"
#include "utility.h"

/**
 * An endian swapping utility
 * Currently undefined
 **/
#define endianSwapIfNeeded(p, nElements, elementSize, swapFlag) {	\
    int  i, j;								\
    char *ptr;								\
    char *c;								\
    char *d;								\
    char e;								\
    if ( swapFlag ) {							\
      ptr = (char *) (p);						\
      for(i=0; i<(nElements); i++,ptr+=(elementSize)) {			\
	c = ptr;							\
	d = c + (elementSize)-1;					\
	for(j=0; j<(elementSize)/2; j++,c++,d--) {			\
	  e = *c;							\
	  *c = *d;							\
	  *d = e;							\
	}								\
      }									\
    }									\
}



struct eampotential_t *myPot = NULL;

struct pmd_base_potential_t *setEamPotFromPotential(struct pmd_base_potential_t *inPot) {
  myPot = (struct eampotential_t *) inPot;
  return (struct pmd_base_potential_t *) myPot;
}
  
struct pmd_base_potential_t *setEamPot(char *dir, char *file) {
  
  if(myPot) eamDestroy((void **) &myPot);
  myPot = eamReadASCII(dir,file);
  myPot->destroy=eamDestroy;
  if ( ! myPot) { printf("Unable to read potential file\n"); exit(1); }
  return (struct pmd_base_potential_t *) myPot;
}

struct eampotential_t *getEamPot() {
  if ( ! myPot) setEamPot((char *) "pots",(char *) "ag");
  myPot->destroy=eamDestroy;
  return myPot;
}

static void destroyPotentialArray(struct potentialarray_t **a, int doubleFlag) {
  if ( ! a ) return;
  if ( ! *a ) return;
  if ( (*a)->values) {
    (*a)->values--;
    (*a)->values-=doubleFlag;
    free((*a)->values);
  }
  free(*a);
  *a = NULL;
  return;
}

static struct potentialarray_t *allocPotentialArray(int n, real_t x0, real_t xn, real_t invDx) {
  struct potentialarray_t *a;
  int is;
  is = (sizeof(struct potentialarray_t)+15 ) & ~15;
  a = (struct potentialarray_t*)malloc(is);
  
  if ( ! a ) return NULL;

  // Always assume double precision arrays!
  is = ((n+3)*sizeof(real_t)+15 ) & ~15;
  a->values = (real_t*)malloc(is);

  if ( ! a->values) {
    free(a);
    return NULL;
  }
  a->values++; 
  a->n = n;
  a->invDx = invDx;
  a->xn = xn;
  a->x0 = x0 + (xn-x0)/(double)n;
  return a;

}


static struct potentialarray_t *getPotentialArrayFromBinaryFile(char *file) {
  struct potentialarray_t *retArray;
  FILE *fp;
  int n;
  int recSize;
  real_t *vals;
  double x0, xn, invDx;
  double *inData; 
  char swapFlag = 0;
  int eightByteHeader = 0;
  int itmp;
  int iflo, ifhi;
  
  fp = fopen(file,"rb");
  if ( ! fp ) {
    return NULL;
  }

  /* read record header and decide swap or not */
  if (fread(&recSize,sizeof(int),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  swapFlag = (recSize > 4096);
  endianSwapIfNeeded(&recSize,1,sizeof(int),swapFlag);

  if (fread(&n,sizeof(int),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&n,1,sizeof(int),swapFlag);

  
  if (fread(&x0,sizeof(double),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&x0,1,sizeof(double),swapFlag);
  
  if (fread(&xn,sizeof(double),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&xn,1,sizeof(double),swapFlag);
  
  if (fread(&invDx,sizeof(double),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&invDx,1,sizeof(double),swapFlag);

  /* discard two integers */
  if (fread(&iflo,sizeof(int),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&iflo,1,sizeof(int),swapFlag);

  if (fread(&ifhi,sizeof(int),1,fp) > 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&ifhi,1,sizeof(int),swapFlag);

  retArray = allocPotentialArray(n,x0,xn,invDx);
  if ( ! retArray ) {
        fclose(fp);
	return NULL;
  }

  /* read record trailer */
  if (fread(&recSize,sizeof(int),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&recSize,1,sizeof(int),swapFlag);
  
  /* read next record header and confirm size */
  if (fread(&recSize,sizeof(int),1,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(&recSize,1,sizeof(int),swapFlag);

  if ( recSize != n*sizeof(double) ) {
    fclose(fp);
    destroyPotentialArray(&retArray,0);
    printf("sizes are: %d,%d\n",recSize, (int)(n*sizeof(double)));
    printf("Size mismatch error reading binary potential file\n");
    exit(1);
  }

  /* allocate space and read in potential data */
  inData = (double*)malloc(n*sizeof(double));
  if ( ! inData) {
    fclose(fp);
    destroyPotentialArray(&retArray,0);
    return NULL;
  }
  if (fread(inData,sizeof(double),n,fp) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  endianSwapIfNeeded(inData,n,sizeof(double),swapFlag);
  vals = retArray->values;
  for(n=0; n<retArray->n; n++, vals++) *vals = (real_t) inData[n];
  free(inData);
  {
    /* array values for robust interpolation */
    real_t lo, hi;
    if(iflo==0) lo=0.0;
    else lo = retArray->values[0];

    if(ifhi==0) hi=0.0;
    else hi = retArray->values[n-1];

    n = retArray->n;
    retArray->values[-1] = lo;
    retArray->values[n] = hi;
    retArray->values[n+1] = hi;

  }
  fclose(fp);
  return retArray;
}


static struct potentialarray_t *getPotentialArrayFromFile(char *file) {
  struct potentialarray_t *retArray;
  char tmp[4096];
  char *ret;
  FILE *fp;
  int n;
  real_t *vals;
  real_t x0, xn, invDx;

  /* check on binary file */
  if(file[strlen(file)-1] != 't') return getPotentialArrayFromBinaryFile(file);

  fp = fopen(file,"r");
  if ( ! fp ) {
    return NULL;
  }

  /**
   * read first line **/
  ret = fgets(tmp,sizeof(tmp),fp);
  sscanf(tmp,"%10d " FMT1 " " FMT1 " " FMT1 ,
	 &n, &x0, &xn, &invDx);

  retArray = allocPotentialArray(n,x0,xn,invDx);
  if ( ! retArray ) return NULL;

  vals = retArray->values;
  for(n=0; n<retArray->n; n++, vals++) 
   if (fscanf(fp,FMT1,vals) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
  {
    /* array values for robust interpolation */
    n = retArray->n;
    retArray->values[-1] = retArray->values[0];
    retArray->values[n] = retArray->values[n-1];
    retArray->values[n+1] = retArray->values[n-1];
  }
  fclose(fp);
  return retArray;
}

static double getMassFromFile(char *file) {
  double mass;
  char tmp[4096];
  FILE *fp;
  char *ret;
  int n;
  fp = fopen(file,"r");
  if ( ! fp ) {
    return -1.0;
  }

  /**
   * read first line **/
  ret = fgets(tmp,sizeof(tmp),fp);
  /**
   * get mass from second line **/
  ret = fgets(tmp,sizeof(tmp),fp);
  sscanf(tmp,"%lf ",&mass);
  return mass;
}

static double getLatFromFile(char *file) {
  double lat;
  char tmp[4096];
  char *ret;
  FILE *fp;
  int n;
  fp = fopen(file,"r");
  if ( ! fp ) {
    return -1.0;
  }

  /**
   * read first two lines **/
  ret = fgets(tmp,sizeof(tmp),fp);
  ret = fgets(tmp,sizeof(tmp),fp);
  /**
   * get lat from third line **/
  ret = fgets(tmp,sizeof(tmp),fp);
  sscanf(tmp,"%lf ",&lat);
  return lat;
}

void eamDestroy(void **inppot) {
  pmd_base_potential_t **pPot = (pmd_base_potential_t **) inppot;
  eampotential_t *pot;
  if ( ! pPot ) return;
  pot = *(eampotential_t **)pPot;
  if ( pot == myPot) myPot = NULL;
  if ( ! pot ) return;
  if(pot->phi) destroyPotentialArray(&(pot->phi),0);
  if(pot->rho) destroyPotentialArray(&(pot->rho),0);
  if(pot->f) destroyPotentialArray(&(pot->f),0);
  free(pot);
  *pPot = NULL;
  myPot = NULL;
  return;
}

eampotential_t *eamReadASCII(char *dir, char *potname) {
  /**
   * reads potential potname from directory dir.
   * returns a poitner to an eampotential_t struct.
   **/
  eampotential_t *retPot;
  char tmp[4096];
  int is;
  is = (sizeof(struct eampotential_t)+15 ) & ~15;
  retPot = (eampotential_t*)malloc(is);
  if ( ! retPot ) return NULL;

  /**
   * read the phi component **/
  sprintf(tmp,"%s/%s.phi",dir,potname);
  retPot->phi = getPotentialArrayFromFile(tmp);

  /**
   * read the rho component **/
  sprintf(tmp,"%s/%s.rho",dir,potname);
  retPot->rho = getPotentialArrayFromFile(tmp);

  /**
   * read the F component **/
  sprintf(tmp,"%s/%s.f",dir,potname);
  retPot->f = getPotentialArrayFromFile(tmp);

  sprintf(tmp,"%s/%s.doc",dir,potname);
  retPot->mass = (real_t) getMassFromFile(tmp);
  retPot->lat = (real_t) getLatFromFile(tmp);

  if ( (retPot->mass < 0.0 ) || (! (retPot->phi && retPot->rho && retPot->f )) ) {
     printf("\n\n"
	    "    ****  Unable to open potential file %s.  **** \n\n"
	    "    Did you untar pots.tgz (tar zxvf pots.tgz)?"
	    "\n\n"
	    ,potname);
     eamDestroy((void **) &retPot);
     
    return NULL;
  }

  /**
   * set the cutoff from the phi component **/
  retPot->cutoff = retPot->phi->xn;

  return retPot;
  
}

/**
 * utility comparison routine **/
static void adiffpot(char *name,potentialarray_t *a, potentialarray_t *b) {
  int i;
  printf("---------------------------------------\n");
  printf("  comparison of %s\n", name);
  printf("    n = %4d   /  %4d\n", a->n, b->n);	 
  printf("   x0 = %10.2g  /  %10.2g\n", a->x0, b->x0);	 
  printf("   xn = %10.2g  /  %10.2g\n", a->xn, b->xn);	 
  printf("   dx = %10.2g  /  %10.2g\n", a->invDx, b->invDx);
  for(i=-1; i<a->n+2;i++) {
    if ( a->values[i] != b->values[i]) {
      printf("   v[%d] = %10.2g  /  %10.2g\n", i, a->values[i],b->values[i]);
    }
  }
  printf("---------------------------------------\n");
  return;
}

void eamComparePots(eampotential_t *a, eampotential_t *b) {
  adiffpot((char *) "phi", a->phi, b->phi);
  adiffpot((char *) "rho", a->rho, b->rho);
  adiffpot((char *) "f", a->f, b->f);
  return;
}

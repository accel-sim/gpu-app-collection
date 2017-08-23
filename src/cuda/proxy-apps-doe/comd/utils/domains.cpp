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
 * this file will split space up into domains so that
 * we will have fast neighbor listing **/

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "domains.h"

#define MAXATOMS 	N_MAX_ATOMS
#define NUMNEIGHBORS	N_MAX_NEIGHBORS

#define freeMe(s,element) {if(s->element) free(s->element);  s->element = NULL;}
void getBoxIxyz(simflat_t *s, int ibox, int *iret) {
    /* given a box id return the ix,iy,iz coordinates */
    iret[0] = ibox%s->nbx[0];
    iret[1] = (ibox/s->nbx[0])%s->nbx[1];
    iret[2] = ibox/s->nbx[0]/s->nbx[1];
    return;
}
static int getBoxIDWorldCoords(simflat_t *s, const real2_t xyz[3], int *ibx) {
    int ibox;
    /* given x,y,z in world co-ordinates return the
       box in which those coordinates fall */

    ibx[0] = (int)(floor(xyz[0]/s->boxsize[0]));
    ibx[1] = (int)(floor(xyz[1]/s->boxsize[1]));
    ibx[2] = (int)(floor(xyz[2]/s->boxsize[2]));

    ibox = ibx[0]+s->nbx[0]*ibx[1]+s->nbx[0]*s->nbx[1]*ibx[2];
    return ibox;
}

void destroyDomains(simflat_t *s) {
    /* release memory for particles */
    freeMe(s,dcenter);
    freeMe(s,natoms);
    freeMe(s,id);
    freeMe(s,iType);
    freeMe(s,mass);
    freeMe(s,r);
    freeMe(s,p);
    freeMe(s,f);
    freeMe(s,fi);
    freeMe(s,rho);
    s->stateflag = SIM_NOSTATE;
    return;
}

static int allocDomainArrays(simflat_t *s) {
    int ierr;
    int i;

    ierr = 0;
    s->dcenter = (real3*)malloc(s->nboxes*sizeof(real3));
    s->natoms = (int*)malloc(s->nboxes*sizeof(int));
    s->id = (int*)malloc(MAXATOMS*s->nboxes*sizeof(int));
    s->iType = (int*)malloc(MAXATOMS*s->nboxes*sizeof(int));
    s->mass = (real_t*)malloc(MAXATOMS*s->nboxes*sizeof(real_t));
    s->r = (real3*)malloc(MAXATOMS*s->nboxes*sizeof(real3));
    s->p = (real3*)malloc(MAXATOMS*s->nboxes*sizeof(real3));
    s->f = (real4*)malloc(MAXATOMS*s->nboxes*sizeof(real4));
    s->fi = (real_t*)malloc(MAXATOMS*s->nboxes*sizeof(real_t));
    s->rho = (real_t*)malloc(MAXATOMS*s->nboxes*sizeof(real_t));

    memset(s->natoms, 0, s->nboxes*sizeof(int));

#ifdef USE_IN_SITU_VIZ
  s->centro = (real_t*)malloc(MAXATOMS*s->nboxes*sizeof(real_t));
#endif 

    ierr += ((s->dcenter)?0:1);
    ierr += ((s->natoms)?0:1);
    ierr += ((s->id)?0:1);
    ierr += ((s->iType)?0:1);
    ierr += ((s->mass)?0:1);
    ierr += ((s->r)?0:1);
    ierr += ((s->p)?0:1);
    ierr += ((s->f)?0:1);
    ierr += ((s->fi)?0:1);
    ierr += ((s->rho)?0:1);

    if(ierr) destroyDomains(s);
    s->stateflag |= SIM_ALLOCED;


    return ierr;

}

static int copyDomainArrays(simflat_t *destination, simflat_t* source) {

    memcpy(destination->dcenter, source->dcenter, source->nboxes*sizeof(real3)); 
    memcpy(destination->natoms, source->natoms, source->nboxes*sizeof(int)); 
    memcpy(destination->id, source->id, MAXATOMS*source->nboxes*sizeof(int)); 
    memcpy(destination->iType, source->iType, MAXATOMS*source->nboxes*sizeof(int)); 
    memcpy(destination->mass, source->mass, MAXATOMS*source->nboxes*sizeof(real_t)); 
    memcpy(destination->r, source->r, MAXATOMS*source->nboxes*sizeof(real3)); 
    memcpy(destination->p, source->p, MAXATOMS*source->nboxes*sizeof(real3)); 
    memcpy(destination->f, source->f, MAXATOMS*source->nboxes*sizeof(real4)); 
    memcpy(destination->fi, source->fi, MAXATOMS*source->nboxes*sizeof(real_t)); 
    memcpy(destination->rho, source->rho, MAXATOMS*source->nboxes*sizeof(real_t)); 

    return 0;
}

void allocDomains(simflat_t *s) {
    int j;
    int ibox;
    if ( ! s ) { printf("s unallocated in allocDomains()\n"); exit(1); }
    if ( ! s->pot) { printf("allocDomains() called without s->pot()"); exit(1); }

    /* decide how many boxes are needed */
    s->nboxes = 1;
    real_t strain[3];
    // dummy strain field
    strain[0] = s->defgrad;
    // temporary kluge to make Viz work
    if (strain[0] == 0.0) strain[0]=1.0;
    strain[1] = 1.0;
    strain[2] = 1.0;
    // box factor
    
    real_t boxfactor = s->bf;
    if (boxfactor == 0.0) boxfactor = 1.0;

    for(j=0; j<3; j++) {
	// 2 halo boxes along each direction
        s->nbx[j] = 2 + (int)floor(s->bounds[j]/(s->pot->cutoff*boxfactor*strain[j]));
	printf("nbx(%d): size = %d, ", j, s->nbx[j]);
        printf("bounds = %e, cutoff = %e, box factor = %e, strain = %e\n",
	 	s->bounds[j], s->pot->cutoff, boxfactor, strain[j]);
        if (s->nbx[j] < 1)  { printf("Need at least one cutoff wide domain in allocDomains()\n"); exit(1);}
        s->nboxes = s->nboxes*s->nbx[j];
        s->boxsize[j] = s->bounds[j]/(real_t)(s->nbx[j]-2);
    }

    if(allocDomainArrays(s)) { printf("Unable to allocate domains in allocDomains()\n"); exit(1); }

    for(ibox=0;ibox<s->nboxes; ibox++) {
        int ib3[3],j;
        getBoxIxyz(s,ibox,ib3);

        for(j=0; j<3; j++) {
            s->dcenter[ibox][j] = s->boxsize[j]*(real_t)ib3[j];
        }
	
	if (ib3[0] == s->nbx[0]-2) s->dcenter[ibox][0] = -s->boxsize[0];
	if (ib3[0] == s->nbx[0]-1) s->dcenter[ibox][0] = s->boxsize[0] *(real_t)(ib3[0]-1);
	if (ib3[1] == s->nbx[1]-2) s->dcenter[ibox][1] = -s->boxsize[1];
	if (ib3[1] == s->nbx[1]-1) s->dcenter[ibox][1] = s->boxsize[1] *(real_t)(ib3[1]-1);
	if (ib3[2] == s->nbx[2]-2) s->dcenter[ibox][2] = -s->boxsize[2];
	if (ib3[2] == s->nbx[2]-1) s->dcenter[ibox][2] = s->boxsize[2] *(real_t)(ib3[2]-1);
    }

    return;
}

void copyDomains(simflat_t *destination, simflat_t *source)
{
    destination->ntot = source->ntot;

    copyDomainArrays(destination, source);
}

void copyAtom(simflat_t *s, int iatom, int ibox, int jatom, int jbox) {
    /* copy atom iatom in domain ibox to atom jatom in box jbox */
    const int ioff = MAXATOMS*ibox+iatom;
    const int joff = MAXATOMS*jbox+jatom;
    s->id[joff] = s->id[ioff];
    s->iType[joff] = s->iType[ioff];
    s->mass[joff] = s->mass[ioff];
    memcpy(s->r[joff],s->r[ioff],sizeof(real3));
    memcpy(s->f[joff],s->f[ioff],sizeof(real4));
    memcpy(s->p[joff],s->p[ioff],sizeof(real3));
    s->fi[joff] = s->fi[ioff];
    s->rho[joff] = s->rho[ioff];
    return;
}

void moveAtom(simflat_t *s, int iId, int iBox, int jBox) {
  int nj,ni;
  nj = s->natoms[jBox];
  copyAtom(s,iId, iBox, nj, jBox);
  s->natoms[jBox]++;
  if(s->natoms[jBox]>= MAXATOMS) { printf("Increase maxatoms\n"); exit(1); }

  s->natoms[iBox]--;
  ni = s->natoms[iBox];
  if(ni) copyAtom(s,ni,iBox,iId,iBox);
  
  return;
}
void putAtomInBox(simflat_t *s,
        const int id, const char flagMove, const int iType,
        const real_t mass,
        const real2_t x,const real2_t y,const real2_t z,
        const real2_t px,const real2_t py,const real2_t pz,
        const real2_t fx,const real2_t fy,const real2_t fz) {

    /**
     * finds an appropriate box for an atom based on the
     * spatial cooridnates and puts it in there.
     *
     * reallocates memory if needed.
     **/
    int ibox;
    int i,m;
    int ioff;
    int ibx[3];
    real2_t xyz[3] = {x,y,z};

    if ( ! s->stateflag ) { printf("ERROR: s not allocated in putAtomInBox()\n"); exit(1); }

    /**
     * push atom into primary period **/
    for(m=0; m<3; m++) {
        if(xyz[m] < 0.0 ) xyz[m] += s->bounds[m];
        else if (xyz[m] >= s->bounds[m] ) xyz[m] -= s->bounds[m];
    }

    /**
     * Find correct box.
     * for now, only one box **/
    ibox = getBoxIDWorldCoords(s,xyz,ibx);
    ioff = ibox*MAXATOMS;
    ioff += s->natoms[ibox];
    /**
     * assign values to array elements **/
    s->ntot++;
    s->natoms[ibox]++;
    s->id[ioff] = id;
    s->iType[ioff] = iType;
    s->mass[ioff] = mass;
    for(m=0; m<3; m++) {
        s->r[ioff][m] = (real_t)(xyz[m]-s->dcenter[ibox][m]);
    }
    s->p[ioff][0] = (real_t)px;
    s->p[ioff][1] = (real_t)py;
    s->p[ioff][2] = (real_t)pz;

    s->f[ioff][0] = (real_t)fx;
    s->f[ioff][1] = (real_t)fy;
    s->f[ioff][2] = (real_t)fz;

    return;
}



/**
 * static box functions **/
static int getIBoxFromIxyz3(simflat_t *s, int *ixyz) {
  int ibox=0;
  int j;

    for(j=0; j<3; j++) {
        if(ixyz[j]<0) ixyz[j] += s->nbx[j];
        else if ( ixyz[j] >= s->nbx[j] ) ixyz[j] -= s->nbx[j];
    }
    ibox = ixyz[0] + ixyz[1]*s->nbx[0] + ixyz[2]*s->nbx[0]*s->nbx[1];

    return ibox;

}
static int getIBoxFromIxyz3NP(simflat_t *s, int *ixyz) {
  int j;

  /**
   * for non-periodic boundary conditions, decide whether we
   * are at a boundary or not **/

  if( ! PERIODIC) {
    for(j=0; j<3; j++) {
      if((ixyz[j]<0)||(ixyz[j]>=s->nbx[j])) return -1; 
    }
  }
  return getIBoxFromIxyz3(s,ixyz);

}
static int getIBoxFromIxyz(simflat_t *s, int ix, int iy, int iz) {
  int ibox;
  int ixyz[3] = {ix,iy,iz};
  return getIBoxFromIxyz3NP(s,ixyz);
}

static int getIBoxFromIxyz_periodic(simflat_t *s, int ix, int iy, int iz) {
  int ixyz[3] = {ix,iy,iz};
int ibox=0;
    for(int j=0; j<3; j++) {
        if(ixyz[j]<0) ixyz[j] = s->nbx[j]-2;
        else if ( ixyz[j] >= s->nbx[j]-2 ) ixyz[j] = s->nbx[j]-1;
    }

    ibox = ixyz[0] + ixyz[1]*s->nbx[0] + ixyz[2]*s->nbx[0]*s->nbx[1];
  return ibox;
}

static void getIxyz3(simflat_t *s, int ibox, int *i3) {
    i3[0] = (ibox%s->nbx[0]);
    i3[1] = (ibox/s->nbx[0])%(s->nbx[1]);
    i3[2] = (ibox/s->nbx[0]/s->nbx[1])%(s->nbx[2]);
    return;
}
static void getIxyz(simflat_t *s, int ibox, int *ix, int *iy, int *iz) {
    int ixyz[3];
    getIxyz3(s,ibox,ixyz);
    *ix = ixyz[0];
    *iy = ixyz[1];
    *iz = ixyz[2];
    return;
}

void update_halos(simflat_t *s)
{
  for (int ibox = 0; ibox < s->nboxes; ibox++) { 
    int ixold[3];
    int ixnew[3];
    int ioff = ibox * MAXATOMS;
    getIxyz3(s, ibox, ixold);
    int jbox;

    ixnew[0] = ixold[0]; 
    ixnew[1] = ixold[1]; 
    ixnew[2] = ixold[2];

    if (ixold[0] == s->nbx[0] - 2) ixnew[0] = s->nbx[0] - 3;
    if (ixold[0] == s->nbx[0] - 1) ixnew[0] = 0;
    if (ixold[1] == s->nbx[1] - 2) ixnew[1] = s->nbx[1] - 3; 
    if (ixold[1] == s->nbx[1] - 1) ixnew[1] = 0; 
    if (ixold[2] == s->nbx[2] - 2) ixnew[2] = s->nbx[2] - 3;
    if (ixold[2] == s->nbx[2] - 1) ixnew[2] = 0;

    if (ixnew[0] == ixold[0] && ixnew[1] == ixold[1] && ixnew[2] == ixold[2]) continue;

      jbox = getIBoxFromIxyz3NP(s, ixnew);
    
    s->natoms[ibox] = s->natoms[jbox];
    for (int i = 0; i < s->natoms[jbox]; i++)
	copyAtom(s, i, jbox, i, ibox);

  }
}

void reBoxAll(simflat_t *s) {
  int ibox;
  for(ibox=0; ibox<s->nboxes; ibox++) {
    int i;
    int ixold[3];
    int ioff;
    getIxyz3(s, ibox, ixold);
    i=0;
    ioff = ibox*MAXATOMS;

    // do not update halos yet
    if (ixold[0] >= s->nbx[0] - 2) continue;
    if (ixold[1] >= s->nbx[1] - 2) continue;
    if (ixold[2] >= s->nbx[2] - 2) continue;

    while(i<s->natoms[ibox]) {
      int ixnew[3];
      int jbox;
      int k;
      real3 rnew;
      memcpy(rnew,s->r[ioff],sizeof(rnew));
      for(k=0; k<3; k++) {
	if(s->r[ioff][k] < 0.0) {
	  ixnew[k] = ixold[k]-1;
	  rnew[k] += s->boxsize[k];
	}
	else if(s->r[ioff][k] >= s->boxsize[k]) {
	  ixnew[k] = ixold[k]+1;
	  rnew[k] -= s->boxsize[k];
	}
	else {
	  ixnew[k] = ixold[k];
	}
      }
      jbox = getIBoxFromIxyz3NP(s,ixnew);
      if((jbox<0)||(jbox==ibox)) {
	/* do nothing if same box or non-periodic boundary */
	i++;
	ioff++;
      }
      else {
	/* move atom to new box */
	memcpy(s->r[ioff],rnew,sizeof(rnew));
	moveAtom(s,i,ibox,jbox);
      }
    }
  }

  update_halos(s);

    return;
}

int *getNeighborBoxes(simflat_t *s, int iBox) {
  /**
   * returns an array whose first element is number of neighbors
   * followed by the ids of those neighbors.
   * the neighbor list is terminated by -1.
   **/
  static int actualNbrs[1+NUMNEIGHBORS];
  int *nbrs;
  int i,j,k;
  int ix, iy, iz;
  int in;

  memset(actualNbrs,-1,1+NUMNEIGHBORS*sizeof(int));
  nbrs = actualNbrs+1;

  if(s->nboxes == 1) {
    nbrs[-1] = 1;
    nbrs[0] = 0;
    return nbrs;
  }

    getIxyz(s, iBox, &ix, &iy, &iz);

    in = 0;

  /* we now get ids from (ix-1,iy-1,iz-1) to (ix+1,iy+1,iz+1)
   * which includes the current box as neighbor 13 */

  for(i=ix-1; i<=ix+1; i++) {
    for(j=iy-1; j<=iy+1; j++) {
      for(k=iz-1; k<=iz+1; k++) {
	nbrs[in] = getIBoxFromIxyz_periodic(s,i,j,k);
	if(nbrs[in] >= 0 ) in++;
      }
    }
  }
   
    nbrs[-1] = in;
    return nbrs;
}


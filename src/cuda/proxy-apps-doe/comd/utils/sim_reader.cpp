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

#include "sim_reader.h"
#include <time.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <float.h>

#define DOREADTIMERS 4

struct simflat_t *blankSimulation(struct pmd_base_potential_t *pot) {
  simflat_t *s;

  s = (simflat_t*)malloc((1*sizeof(simflat_t)));
  memset(s, 0, sizeof(simflat_t));

  if(pot) {
    if ( pot->cutoff < 0.0) { printf("Blanksimulation() got a negative cutoff\n"); exit(1); }
    s->pot = pot;
  }
  else {
    extern ljpotential_t *getLJPot();
    s->pot = (pmd_base_potential_t *) getLJPot();
  }

  s->stateflag = 0;


  return s;

}

simflat_t *create_fcc_lattice(command_t cmd, struct pmd_base_potential_t *pot) {
    /**
     * Creates an fcc lattice with nx * ny * nz unit cells and lattice constant lat
     *
     **/
    int nx = cmd.nx;
    int ny = cmd.ny;
    int nz = cmd.nz;
    real_t lat = cmd.lat;
    real_t defgrad = cmd.defgrad;
    real_t boxfactor = cmd.bf;

    simflat_t *s = NULL;
    int     i, j, k, n, itype, natoms;
    real_t  x, y, z, halflat;
    real2_t fx,fy,fz, px,py,pz;
    fx=fy=fz=px=py=pz=0.0;
#ifdef DOICTIMERS  
    clock_t start,old,count=0;
#endif

#ifdef DOICTIMERS  
    start = clock();
#endif
    s = blankSimulation(pot);
    if ( ! s ) { printf("Unable to create Simulation data structure\n"); exit(1); }

    /* Optional simulation comment (blank for now) */
    s->comment = (char*)malloc(1024*sizeof(char));

    natoms = 4*nx*ny*nz;
    halflat = lat / 2.0;

    /* periodic  boundaries  */
    memset(s->bounds,0,sizeof(real3));
    memset(s->boxsize,0,sizeof(real3));
    // stretch in x direction
    s->defgrad = defgrad;
    s->bounds[0] = nx * lat * defgrad;
    s->bounds[1] = ny * lat;
    s->bounds[2] = nz * lat;

    s->bf = boxfactor;
#if DOICTIMERS  >2
    old = clock();
#endif
    allocDomains(s);
#if DOICTIMERS  >2
    count = clock()-old+count;
#endif
    i = j = k = n = 0;
    z = lat / 4.0;
    while (z < s->bounds[2]) {
	y = lat / 4.0;
	while (y < s->bounds[1]) {
	    x = lat * defgrad / 4.0;
	    while (x < s->bounds[0]) {
		if ((i+j+k) % 2)
		    putAtomInBox(s,n++,1,1,s->pot->mass,x,y,z,px,py,pz,fx,fy,fz);
		x += halflat * defgrad;
		i++;
	    }
	    y += halflat;
	    j++;
	}
	z += halflat;
	k++;
    }

#ifdef DOICTIMERS  
    old = clock();
    if(DOICTIMERS) printf("\n    ---- FCC initial condition took %.2gs for %d atoms\n\n",
	    (float)(old-start)/(float)(CLOCKS_PER_SEC), n);
#endif

    return s;
}

#ifndef GZIPSUPPORT
simflat_t *fromFileASCII(command_t cmd, struct pmd_base_potential_t *pot) {
    /**
     * Reads in a simulation from a file written by clsman.
     *
     * Initially only mode 0 files are read
     **/
    // assign the needed parameters from the commnad struct

    char *filename = cmd.filename;
    char *ret;
    simflat_t *s = NULL;
    FILE *fp;
    int i,itype;
    real2_t x,y,z,px,py,pz;
    int natoms, nmove;
#ifdef DOREADTIMERS  
    clock_t start,old;
    clock_t count=0;
    real2_t fx,fy,fz;
    fx=fy=fz=0.0;
#endif

    fp = fopen(filename,"r");
    if( ! fp ) return NULL;

#ifdef DOREADTIMERS  
    start = clock();
#endif
    s = blankSimulation(pot);
    if ( ! s ) {
        fclose(fp);
        return s;
    }

    s->bf = cmd.bf;

    /* read in natom and nmove */
    if (fscanf(fp,"%d%d\n",&natoms,&nmove) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
    PMDDEBUGPRINTF(0,"\n   natoms = %d\n   nmove = %d\n\n",natoms,nmove);
    /* read in comment */
    s->comment = (char*)malloc(1024*sizeof(char));
    if ( nmove) {
        ret = fgets(s->comment,1024,fp);
    }

    /* read in periodic  boundaries */
    memset(s->bounds,0,sizeof(real3));
    memset(s->boxsize,0,sizeof(real3));

    if (fscanf(fp,FMT1 " " FMT1 " " FMT1 "\n",s->bounds,s->bounds+1,s->bounds+2) < 1) fprintf(stderr, "\nError in reading or end of file.\n");

#if DOREADTIMERS  >2
    old = clock();
#endif
    allocDomains(s);
#if DOREADTIMERS  >2
    count = clock()-old+count;
#endif
    for(i=0; i<natoms; i++) {
        PMDDEBUGPRINTF(0,"Reading %d\r", i);
        if (fscanf(fp,FMT2 " " FMT2 " " FMT2 " %d " FMT2 " " FMT2 " " FMT2 "\n",
                &x,&y,&z,&itype,&px,&py,&pz) < 1) fprintf(stderr, "\nError in reading or end of file.\n");
        PMDDEBUGPRINTF(0,"%4d: " FMT2 " " FMT2 " " FMT2 " %d " FMT2 " " FMT2 " " FMT2 "\n",
                i,x,y,z,itype,px,py,pz);
#if DOREADTIMERS  >2
        old = clock();
#endif
        putAtomInBox(s,i,1,1,s->pot->mass,
                x,y,z,px,py,pz,fx,fy,fz);
#if DOREADTIMERS  >2
        count = clock()-old+count;
#endif
    }
#ifdef DOREADTIMERS  
    old = clock();
#endif
    fclose(fp);


    return s;
}


simflat_t *fromFileGzip(command_t cmd, struct pmd_base_potential_t *pot) {
    printf("    fromFileGzip(): Trying fromFileASCII()n");
    return (fromFileASCII(cmd,pot));
}

#else
#include "zlib.h"
simflat_t *fromFileGzip(command_t cmd, struct pmd_base_potential_t *pot) {
    /**
     * Reads in a simulation from a file written by clsman.
     *
     * Initially only mode 0 files are read
     **/
#ifdef DOREADTIMERS  
    clock_t start,old;
    int count;
#endif
    char *filename = cmd.filename;
    simflat_t *s = NULL;
    gzFile fp;
    int i,itype;
    real2_t x,y,z,px,py,pz;
    int natoms, nmove;
    char zstr[16348];
    real2_t fx,fy,fz;
    fx=fy=fz=0.0;
#ifdef DOREADTIMERS  
    start = clock();
    count = 0;
#endif
    PMDDEBUGPRINTF(0,"tmp file in read is: %s\n", filename);
    fp = gzopen(filename,"rb");
    if( ! fp ) return NULL;

    s = blankSimulation(pot);
    if ( ! s ) {
        gzclose(fp);
        return s;
    }
    s->bf = cmd.bf;


    /* read in natom and nmove */
    gzgets(fp,zstr,sizeof(zstr));
    sscanf(zstr,"%d%d\n",&natoms,&nmove);
    PMDDEBUGPRINTF(0,"\n   -0natoms = %d\n   nmove = %d\n\n",natoms,nmove);

    /* read in comment */
    s->comment = (char*)malloc(1024*sizeof(char));
    if(0) {
        char fmta[128];
        sprintf(fmta,"%%%d\n",nmove);
        gzgets(fp,zstr,sizeof(zstr));
        sscanf(zstr,fmta,s->comment);
    }
    else {
        gzgets(fp,zstr,sizeof(zstr));
        if(1)strncpy(s->comment,zstr,1024);
        else s->comment = (char *) "no comment";
    }

    PMDDEBUGPRINTF(0,"\n   comment = %s\n",s->comment);

    /* read in periodic  boundaries */
    gzgets(fp,zstr,sizeof(zstr));
    sscanf(zstr,FMT1 " " FMT1 " " FMT1 "\n",s->bounds,s->bounds+1,s->bounds+2);
    PMDDEBUGPRINTF(0,"Periodic bounds: " FMT1 " " FMT1 " " FMT1 "\n",
            s->bounds[0],s->bounds[1],s->bounds[2]);

    allocDomains(s);

    for(i=0; i<natoms; i++) {
        gzgets(fp,zstr,sizeof(zstr));
        sscanf(zstr,FMT2 " " FMT2 " " FMT2 " %d " FMT2 " " FMT2 " " FMT2 "\n",
                &x,&y,&z,&itype,&px,&py,&pz);
        PMDDEBUGPRINTF(0,"%4d: " FMT2 " " FMT2 " " FMT2 " %d " FMT2 " " FMT2 " " FMT2 "\n",
                i,x,y,z,itype,px,py,pz);
#ifdef DOREADTIMERS  
        old = clock();
#endif
        putAtomInBox(s,i,1,0,s->pot->mass,
                x,y,z,px,py,pz,fx,fy,fz);
#ifdef DOREADTIMERS  
        count = clock()-old+count;
#endif
    }
    gzclose(fp);
#ifdef DOREADTIMERS  
    old=clock();
#endif
    return s;
}

simflat_t *fromFileASCII(command_t cmd, struct pmd_base_potential_t *pot) {
    return(fromFileGzip(cmd,pot));
}


simflat_t *fromFileTim(command_t cmd, struct pmd_base_potential_t *pot) {
    char *filename = cmd.filename;
    simflat_t *s = NULL;
    FILE *fp;
    int i,itype;
    real2_t x,y,z,px,py,pz;
    int natoms, nmove;

    fp = fopen(filename,"r");
    if( ! fp ) return NULL;
    fclose(fp);

    s = blankSimulation(pot);
    if ( ! s ) {
        fclose(fp);
        return s;
    }

    struct stat filestatus;
    stat( filename, &filestatus );
    natoms = filestatus.st_size/sizeof(FileAtom);

    fp = fopen(filename,"r");
    if( ! fp ) return NULL;

    FileAtom* atoms = (FileAtom*)malloc(natoms*sizeof(FileAtom));
    size_t n = fread((void*)atoms, sizeof(FileAtom), natoms, fp); 
    printf("Number of atoms: %d %d\n", (int)natoms, (int)n);
    fclose(fp);

    float minX, minY, minZ, maxX, maxY, maxZ;
    minX = minY = minZ = FLT_MAX;
    maxX = maxY = maxZ = FLT_MIN;
    for(i = 0; i < natoms; ++i) 
    {
        if (atoms[i].x < minX) minX = atoms[i].x;
        if (atoms[i].x > maxX) maxX = atoms[i].x;
        if (atoms[i].y < minY) minY = atoms[i].y;
        if (atoms[i].y > maxY) maxY = atoms[i].y;
        if (atoms[i].z < minZ) minZ = atoms[i].z;
        if (atoms[i].z > maxZ) maxZ = atoms[i].z;
    }

    minX = minY = minZ = 0.0;
    maxX = 327.42399999999997817;
    maxY = 354.44687726089500757;
    maxZ = 3007.58148577889733133;

    memset(s->bounds,0,sizeof(real3));
    memset(s->boxsize,0,sizeof(real3));
    s->bounds[0] = maxX;  s->bounds[1] = maxY;  s->bounds[2] = maxZ;  

    s->bf = cmd.bf;

    allocDomains(s);

    for(i=0; i<natoms; i++) 
    {
        putAtomInBox(s, i, 1, 1, s->pot->mass, atoms[i].x, atoms[i].y, atoms[i].z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    free(atoms);

    printf("Finished reading\n");
    printf("Bounds: %f %f %f\n", s->bounds[0],s->bounds[1],s->bounds[2]);

    return s;
}


#endif




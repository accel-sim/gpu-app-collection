/// \file
/// CoMD data structures.

#ifndef __COMDTYPES_H_
#define __COMDTYPES_H_

#include <stdio.h>
#include "mytype.h"
#include "haloExchange.h"
#include "linkCells.h"
#include "decomposition.h"
#include "initAtoms.h"
#include "gpu_types.h"

// windows 
#if defined(_WIN32) || defined(_WIN64) 
  #define snprintf _snprintf 
  #define vsnprintf _vsnprintf 
  #define strcasecmp _stricmp 
  #define strncasecmp _strnicmp 
  // use c++ for VS 2010
  #define EXTERN_C extern "C"
#else
  // use c99 for linux
  #define EXTERN_C extern
#endif

struct SimFlatSt;

/// The base struct from which all potentials derive.  Think of this as an
/// abstract base class.
///
/// CoMD uses the following units:
///  - distance is in Angstroms
///  - energy is in eV
///  - time is in fs
///  - force in in eV/Angstrom
///
///  The choice of distance, energy, and time units means that the unit
///  of mass is eV*fs^2/Angstrom^2.  Hence, we must convert masses that
///  are input in AMU (atomic mass units) into internal mass units.
typedef struct BasePotentialSt 
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t mass;            //!< mass of atoms in intenal units
   real_t lat;             //!< lattice spacing (angs) of unit cell
   char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
   char  name[3];	   //!< element name
   int	 atomicNo;	   //!< atomic number  
   int  (*force)(struct SimFlatSt* s); //!< function pointer to force routine
   void (*print)(FILE* file, struct BasePotentialSt* pot);
   void (*destroy)(struct BasePotentialSt** pot); //!< destruction of the potential
} BasePotential;

/// species data: chosen to match the data found in the setfl/funcfl files
typedef struct SpeciesDataSt
{
   char  name[3];   //!< element name
   int	 atomicNo;  //!< atomic number  
   real_t mass;     //!< mass in internal units
} SpeciesData;

/// Simple struct to store the initial energy and number of atoms. 
/// Used to check energy conservation and atom conservation. 
typedef struct ValidateSt
{
   double eTot0; //<! Initial total energy
   int nAtoms0;  //<! Initial global number of atoms
} Validate;

/// 
/// The fundamental simulation data structure with MAXATOMS in every
/// link cell.
/// 
typedef struct SimFlatSt
{
   int nSteps;            //<! number of time steps to run
   int printRate;         //<! number of steps between output
   double dt;             //<! time step
   
   Domain* domain;        //<! domain decomposition data

   LinkCell* boxes;       //<! link-cell data

   Atoms* atoms;          //<! atom data (positions, momenta, ...)

   SpeciesData* species;  //<! species data (per species, not per atom)
   
   real_t ePotential;     //!< the total potential energy of the system
   real_t eKinetic;       //!< the total kinetic energy of the system

   BasePotential *pot;	  //!< the potential

   HaloExchange* atomExchange;
   
   SimGpu host;		// host data for staging
   SimGpu gpu;		// GPU data

   // latency hiding
   int n_boundary_cells;	// note that this is 2-size boundary to allow overlap for atoms exchange
   int *boundary_cells;		
   int *interior_cells;

   int n_boundary1_cells;	// 1-size boundary - necessary for sorting only
   int *boundary1_cells;	// boundary cells that are neighbors to halos

   // streams for async execution
   cudaStream_t boundary_stream;	
   cudaStream_t interior_stream;

   // gpu options
   int gpuAsync;
   int gpuProfile;
    
   int *flags;			// flags for renumbering
   int *tmp_sort;		// temp array for merge sort
   char *gpu_atoms_buf;		// buffer for atoms exchange
   char *gpu_force_buf;		// buffer for forces exchange
   
} SimFlat;

/// Derived struct for a Lennard Jones potential.
/// Polymorphic with BasePotential.
/// \see BasePotential
typedef struct LjPotentialSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t mass;            //!< mass of atoms in intenal units
   real_t lat;             //!< lattice spacing (angs) of unit cell
   char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
   char  name[3];          //!< element name
   int   atomicNo;         //!< atomic number  
   int  (*force)(SimFlat* s); //!< function pointer to force routine
   void (*print)(FILE* file, BasePotential* pot);
   void (*destroy)(BasePotential** pot); //!< destruction of the potential
   real_t sigma;
   real_t epsilon;
} LjPotential;

/// Pointers to the data that is needed in the load and unload functions
/// for the force halo exchange.
/// \see loadForceBuffer
/// \see unloadForceBuffer
typedef struct ForceExchangeDataSt
{
   real_t* dfEmbed; //<! derivative of embedding energy
   struct LinkCellSt* boxes;
}ForceExchangeData;

/// Handles interpolation of tabular data.
///
/// \see initInterpolationObject
/// \see interpolate
typedef struct InterpolationObjectSt
{
   int n;          //!< the number of values in the table
   real_t x0;      //!< the starting ordinate range
   real_t invDx;   //!< the inverse of the table spacing
   real_t* values; //!< the abscissa values
} InterpolationObject;

/// Derived struct for an EAM potential.
/// Uses table lookups for function evaluation.
/// Polymorphic with BasePotential.
/// \see BasePotential
typedef struct EamPotentialSt
{
   real_t cutoff;          //!< potential cutoff distance in Angstroms
   real_t mass;            //!< mass of atoms in intenal units
   real_t lat;             //!< lattice spacing (angs) of unit cell
   char latticeType[8];    //!< lattice type, e.g. FCC, BCC, etc.
   char  name[3];          //!< element name
   int   atomicNo;         //!< atomic number  
   int  (*force)(SimFlat* s); //!< function pointer to force routine
   void (*print)(FILE* file, BasePotential* pot);
   void (*destroy)(BasePotential** pot); //!< destruction of the potential
   InterpolationObject* phi;  //!< Pair energy
   InterpolationObject* rho;  //!< Electron Density
   InterpolationObject* f;    //!< Embedding Energy

   real_t* rhobar;        //!< per atom storage for rhobar
   real_t* dfEmbed;       //!< per atom storage for derivative of Embedding
   HaloExchange* forceExchange;
   ForceExchangeData* forceExchangeData;
} EamPotential;

#endif

/// \file
/// Compute forces for the Embedded Atom Model (EAM).
///
/// The Embedded Atom Model (EAM) is a widely used model of atomic
/// interactions in simple metals.
/// 
/// http://en.wikipedia.org/wiki/Embedded_atom_model
///
/// In the EAM, the total potential energy is written as a sum of a pair
/// potential and the embedding energy, F:
///
/// \f[
///   U = \sum_{ij} \varphi(r_{ij}) + \sum_i F({\bar\rho_i})
/// \f]
///
/// The pair potential \f$\varphi_{ij}\f$ is a two-body inter-atomic
/// potential, similar to the Lennard-Jones potential, and
/// \f$F(\bar\rho)\f$ is interpreted as the energy required to embed an
/// atom in an electron field with density \f$\bar\rho\f$.  The local
/// electon density at site i is calulated by summing the "effective
/// electron density" due to all neighbors of atom i:
///
/// \f[
/// \bar\rho_i = \sum_j \rho_j(r_{ij})
/// \f]
///
/// The force on atom i, \f${\bf F}_i\f$ is given by
///
/// \f{eqnarray*}{
///   {\bf F}_i & = & -\nabla_i \sum_{jk} U(r_{jk})\\
///       & = & - \sum_j\left\{
///                  \varphi'(r_{ij}) +
///                  [F'(\bar\rho_i) + F'(\bar\rho_j)]\rho'(r_{ij})
///                \right\} \hat{r}_{ij}
/// \f}
///
/// where primes indicate the derivative of a function with respect to
/// its argument and \f$\hat{r}_{ij}\f$ is a unit vector in the
/// direction from atom i to atom j.
///
/// The form of this force expression has two significant consequences.
/// First, unlike with a simple pair potential, it is not possible to
/// compute the potential energy and the forces on the atoms in a single
/// loop over the pairs.  The terms involving \f$ F'(\bar\rho) \f$
/// cannot be calculated until \f$ \bar\rho \f$ is known, but
/// calculating \f$ \bar\rho \f$ requires a loop over the pairs.  Hence
/// the EAM force routine contains three loops.
///
///   -# Loop over all pairs, compute the two-body
///   interaction and the electron density at each atom
///   -# Loop over all atoms, compute the embedding energy and its
///   derivative for each atom
///   -# Loop over all pairs, compute the embedding
///   energy contribution to the force and add to the two-body force
///
/// The second loop over pairs doubles the data motion requirement
/// relative to a simple pair potential.
///
/// The second consequence of the force expression is that computing the
/// forces on all atoms requires additional communication beyond the
/// coordinates of all remote atoms within the cutoff distance.  This is
/// again because of the terms involving \f$ F'(\bar\rho_j) \f$.  If
/// atom j is a remote atom, the local task cannot compute \f$
/// \bar\rho_j \f$.  (Such a calculation would require all the neighbors
/// of atom j, some of which can be up to 2 times the cutoff distance
/// away from a local atom---outside the typical halo exchange range.)
///
/// To obtain the needed remote density we introduce a second halo
/// exchange after loop number 2 to communicate \f$ F'(\bar\rho) \f$ for
/// remote atoms.  This provides the data we need to complete the third
/// loop, but at the cost of introducing a communication operation in
/// the middle of the force routine.
///
/// At least two alternate methods can be used to deal with the remote
/// density problem.  One possibility is to extend the halo exchange
/// radius for the atom exchange to twice the potential cutoff distance.
/// This is likely undesirable due to large increase in communication
/// volume.  The other possibility is to accumulate partial force terms
/// on the tasks where they can be computed.  In this method, tasks will
/// compute force contributions for remote atoms, then communicate the
/// partial forces at the end of the halo exchange.  This method has the
/// advantage that the communication is deffered until after the force
/// loops, but the disadvantage that three times as much data needs to
/// be set (three components of the force vector instead of a single
/// scalar \f$ F'(\bar\rho) \f$.


#include "eam.h"

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

#include "constants.h"
#include "memUtils.h"
#include "parallel.h"
#include "linkCells.h"
#include "CoMDTypes.h"
#include "performanceTimers.h"
#include "haloExchange.h"

EXTERN_C void updateNeighborsGpu(SimGpu sim, int *temp);
EXTERN_C void updateNeighborsGpuAsync(SimGpu sim, int *temp, int num_cells, int *cell_list, cudaStream_t stream);

EXTERN_C void eamForce1Gpu(SimGpu sim);
EXTERN_C void eamForce2Gpu(SimGpu sim);
EXTERN_C void eamForce3Gpu(SimGpu sim);

// latency hiding opt
EXTERN_C void eamForce1GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream);
EXTERN_C void eamForce2GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream);
EXTERN_C void eamForce3GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream);

#define MAX(A,B) ((A) > (B) ? (A) : (B))


// EAM functionality
static int eamForce(SimFlat* s);
static void eamPrint(FILE* file, BasePotential* pot);
static void eamDestroy(BasePotential** pot); 
static void eamBcastPotential(EamPotential* pot);


// Table interpolation functionality
static InterpolationObject* initInterpolationObject(
   int n, real_t x0, real_t dx, real_t* data);
static void destroyInterpolationObject(InterpolationObject** table);
static void interpolate(InterpolationObject* table, real_t r, real_t* f, real_t* df);
static void bcastInterpolationObject(InterpolationObject** table);
static void printTableData(InterpolationObject* table, const char* fileName);


// Read potential tables from files.
static void eamReadSetfl(EamPotential* pot, const char* dir, const char* potName);
static void eamReadFuncfl(EamPotential* pot, const char* dir, const char* potName);
static void fileNotFound(const char* callSite, const char* filename);
static void notAlloyReady(const char* callSite);
static void typeNotSupported(const char* callSite, const char* type);


/// Allocate and initialize the EAM potential data structure.
///
/// \param [in] dir   The directory in which potential table files are found.
/// \param [in] file  The name of the potential table file.
/// \param [in] type  The file format of the potential file (setfl or funcfl).
BasePotential* initEamPot(const char* dir, const char* file, const char* type)
{
   EamPotential* pot = (EamPotential*)comdMalloc(sizeof(EamPotential));
   assert(pot);
   pot->force = eamForce;
   pot->print = eamPrint;
   pot->destroy = eamDestroy;
   pot->phi = NULL;
   pot->rho = NULL;
   pot->f   = NULL;

   // Initialization of the next three items requires information about
   // the parallel decomposition and link cells that isn't available
   // with the potential is initialized.  Hence, we defer their
   // initialization until the first time we call the force routine.
   pot->dfEmbed = NULL;
   pot->rhobar  = NULL;
   pot->forceExchange = NULL;

   if (getMyRank() == 0)
   {
      if (strcmp(type, "setfl" ) == 0)
         eamReadSetfl(pot, dir, file);
      else if (strcmp(type,"funcfl") == 0)
         eamReadFuncfl(pot, dir, file);
      else
         typeNotSupported("initEamPot", type);
   }
   eamBcastPotential(pot);
   
   return (BasePotential*) pot;
}

/// Calculate potential energy and forces for the EAM potential.
///
/// Three steps are required:
///
///   -# Loop over all atoms and their neighbors, compute the two-body
///   interaction and the electron density at each atom
///   -# Loop over all atoms, compute the embedding energy and its
///   derivative for each atom
///   -# Loop over all atoms and their neighbors, compute the embedding
///   energy contribution to the force and add to the two-body force
/// 
int eamForce(SimFlat* s)
{
   EamPotential* pot = (EamPotential*) s->pot;
   assert(pot);

   if (s->gpuAsync) {   
     // only update neighbors list when method != thread_atom
     if (s->gpu.method > 1) 
       updateNeighborsGpuAsync(s->gpu, s->flags, s->n_boundary_cells, s->boundary_cells, s->boundary_stream);

     // interior stream already launched
     eamForce1GpuAsync(s->gpu, s->gpu.b_list, s->boundary_stream);
     eamForce2GpuAsync(s->gpu, s->gpu.b_list, s->boundary_stream);

     // we need boundary data before halo exchange
     cudaStreamSynchronize(s->boundary_stream);

     // now we can start step 3 on the interior
     eamForce3GpuAsync(s->gpu, s->gpu.i_list, s->interior_stream);
   }
   else {
     // only update neighbors list when method != thread_atom
     if (s->gpu.method > 1) 
       updateNeighborsGpu(s->gpu, s->flags);

     eamForce1Gpu(s->gpu);

     if (!s->gpuProfile)
       eamForce2Gpu(s->gpu);
   }
 
   if (!s->gpuProfile) {
     // exchange derivative of the embedding energy with repsect to rhobar
     startTimer(eamHaloTimer);
     haloExchange(pot->forceExchange, s);
     stopTimer(eamHaloTimer);

     if (s->gpuAsync) {
       // we need updated interior data before 3rd step
       cudaStreamSynchronize(s->interior_stream);
     }
 
     if (s->gpuAsync) {
       // interior stream already launched
       eamForce3GpuAsync(s->gpu, s->gpu.b_list, s->boundary_stream);
     }
     else {
       eamForce3Gpu(s->gpu);
     }
   }

   return 0;
}

void eamPrint(FILE* file, BasePotential* pot)
{
   EamPotential *eamPot = (EamPotential*) pot;
   fprintf(file, "  Potential type  : EAM\n");
   fprintf(file, "  Species name    : %s\n", eamPot->name);
   fprintf(file, "  Atomic number   : %d\n", eamPot->atomicNo);
   fprintf(file, "  Mass            : "FMT1" amu\n", eamPot->mass/amuToInternalMass); // print in amu
   fprintf(file, "  Lattice type    : %s\n", eamPot->latticeType);
   fprintf(file, "  Lattice spacing : "FMT1" Angstroms\n", eamPot->lat);
   fprintf(file, "  Cutoff          : "FMT1" Angstroms\n", eamPot->cutoff);
}

void eamDestroy(BasePotential** pPot)
{
   if ( ! pPot ) return;
   EamPotential* pot = *(EamPotential**)pPot;
   if ( ! pot ) return;
   destroyInterpolationObject(&(pot->phi));
   destroyInterpolationObject(&(pot->rho));
   destroyInterpolationObject(&(pot->f));
   destroyHaloExchange(&(pot->forceExchange));
   comdFree(pot);
   *pPot = NULL;

   return;
}

/// Broadcasts an EamPotential from rank 0 to all other ranks.
/// If the table coefficients are read from a file only rank 0 does the
/// read.  Hence we need to broadcast the potential to all other ranks.
void eamBcastPotential(EamPotential* pot)
{
   assert(pot);
   
   struct 
   {
      real_t cutoff, mass, lat;
      char latticeType[8];
      char name[3];
      int atomicNo;
   } buf;
   if (getMyRank() == 0)
   {
      buf.cutoff   = pot->cutoff;
      buf.mass     = pot->mass;
      buf.lat      = pot->lat;
      buf.atomicNo = pot->atomicNo;
      strcpy(buf.latticeType, pot->latticeType);
      strcpy(buf.name, pot->name);
   }
   bcastParallel(&buf, sizeof(buf), 0);
   pot->cutoff   = buf.cutoff;
   pot->mass     = buf.mass;
   pot->lat      = buf.lat;
   pot->atomicNo = buf.atomicNo;
   strcpy(pot->latticeType, buf.latticeType);
   strcpy(pot->name, buf.name);

   bcastInterpolationObject(&pot->phi);
   bcastInterpolationObject(&pot->rho);
   bcastInterpolationObject(&pot->f);
}

/// Builds a structure to store interpolation data for a tabular
/// function.  Interpolation must be supported on the range
/// \f$[x_0, x_n]\f$, where \f$x_n = n*dx\f$.
///
/// \see interpolate
/// \see bcastInterpolationObject
/// \see destroyInterpolationObject
///
/// \param [in] n    number of values in the table.
/// \param [in] x0   minimum ordinate value of the table.
/// \param [in] dx   spacing of the ordinate values.
/// \param [in] data abscissa values.  An array of size n. 
InterpolationObject* initInterpolationObject(
   int n, real_t x0, real_t dx, real_t* data)
{
   InterpolationObject* table =
      (InterpolationObject *)comdMalloc(sizeof(InterpolationObject)) ;
   assert(table);

   table->values = (real_t*)comdCalloc(1, (n+3)*sizeof(real_t));
   assert(table->values);

   table->values++; 
   table->n = n;
   table->invDx = 1.0/dx;
   table->x0 = x0;

   for (int ii=0; ii<n; ++ii)
      table->values[ii] = data[ii];
   
   table->values[-1] = table->values[0];
   table->values[n+1] = table->values[n] = table->values[n-1];

   return table;
}

void destroyInterpolationObject(InterpolationObject** a)
{
   if ( ! a ) return;
   if ( ! *a ) return;
   if ( (*a)->values)
   {
      (*a)->values--;
      comdFree((*a)->values);
   }
   comdFree(*a);
   *a = NULL;

   return;
}

/// Broadcasts an InterpolationObject from rank 0 to all other ranks.
///
/// It is commonly the case that the data needed to create the
/// interpolation table is available on only one task (for example, only
/// one task has read the data from a file).  Broadcasting the table
/// eliminates the need to put broadcast code in multiple table readers.
///
/// \see eamBcastPotential
void bcastInterpolationObject(InterpolationObject** table)
{
   struct
   {
      int n;
      real_t x0, invDx;
   } buf;

   if (getMyRank() == 0)
   {
      buf.n     = (*table)->n;
      buf.x0    = (*table)->x0;
      buf.invDx = (*table)->invDx;
   }
   bcastParallel(&buf, sizeof(buf), 0);

   if (getMyRank() != 0)
   {
      assert(*table == NULL);
      *table = (InterpolationObject*)comdMalloc(sizeof(InterpolationObject));
      (*table)->n      = buf.n;
      (*table)->x0     = buf.x0;
      (*table)->invDx  = buf.invDx;
      (*table)->values = (real_t*)comdMalloc(sizeof(real_t) * (buf.n+3) );
      (*table)->values++;
   }
   
   int valuesSize = sizeof(real_t) * ((*table)->n+3);
   bcastParallel((*table)->values-1, valuesSize, 0);
}

void printTableData(InterpolationObject* table, const char* fileName)
{
   if (!printRank()) return;

   FILE* potData;
   potData = fopen(fileName,"w");
   real_t dR = 1.0/table->invDx;
   for (int i = 0; i<table->n; i++)
   {
      real_t r = table->x0+i*dR;
      fprintf(potData, "%d %e %e\n", i, r, table->values[i]);
   }
   fclose(potData);
}
   
/// Reads potential data from a setfl file and populates
/// corresponding members and InterpolationObjects in an EamPotential.
///
/// setfl is a file format for tabulated potential functions used by
/// the original EAM code DYNAMO.  A setfl file contains EAM
/// potentials for multiple elements.
///
/// The contents of a setfl file are:
///
/// | Line Num | Description
/// | :------: | :----------
/// | 1 - 3    | comments
/// | 4        | ntypes type1 type2 ... typen
/// | 5        | nrho     drho     nr   dr   rcutoff
/// | F, rho   | Following line 5 there is a block for each atom type with F, and rho.
/// | b1       | ielem(i)   amass(i)     latConst(i)    latType(i)
/// | b2       | embedding function values F(rhobar) starting at rhobar=0
/// |   ...    | (nrho values. Multiple values per line allowed.)
/// | bn       | electron density, starting at r=0
/// |   ...    | (nr values. Multiple values per line allowed.)
/// | repeat   | Return to b1 for each atom type.
/// | phi      | phi_ij for (1,1), (2,1), (2,2), (3,1), (3,2), (3,3), (4,1), ..., 
/// | p1       | pair potential between type i and type j, starting at r=0
/// |   ...    | (nr values. Multiple values per line allowed.)
/// | repeat   | Return to p1 for each phi_ij
///
/// Where:
///    -  ntypes        :      number of element types in the potential  
///    -  nrho          :      number of points the embedding energy F(rhobar)
///    -  drho          :      table spacing for rhobar 
///    -  nr            :      number of points for rho(r) and phi(r)
///    -  dr            :      table spacing for r in Angstroms
///    -  rcutoff       :      cut-off distance in Angstroms
///    -  ielem(i)      :      atomic number for element(i)
///    -  amass(i)      :      atomic mass for element(i) in AMU
///    -  latConst(i)   :      lattice constant for element(i) in Angstroms
///    -  latType(i)    :      lattice type for element(i)  
///
/// setfl format stores r*phi(r), so we need to converted to the pair
/// potential phi(r).  In the file, phi(r)*r is in eV*Angstroms.
/// NB: phi is not defined for r = 0
///
/// F(rhobar) is in eV.
///
void eamReadSetfl(EamPotential* pot, const char* dir, const char* potName)
{
   char tmp[4096];
#if defined(_WIN32) || defined(_WIN64)
   sprintf(tmp, "%s\\%s", dir, potName);
#else
   sprintf(tmp, "%s/%s", dir, potName);
#endif

   FILE* potFile = fopen(tmp, "r");
   if (potFile == NULL)
      fileNotFound("eamReadSetfl", tmp);
   
   // read the first 3 lines (comments)
   fgets(tmp, sizeof(tmp), potFile);
   fgets(tmp, sizeof(tmp), potFile);
   fgets(tmp, sizeof(tmp), potFile);

   // line 4
   fgets(tmp, sizeof(tmp), potFile);
   int nElems;
   sscanf(tmp, "%d", &nElems);
   if( nElems != 1 )
      notAlloyReady("eamReadSetfl");

   //line 5
   int nRho, nR;
   double dRho, dR, cutoff;
   //  The same cutoff is used by all alloys, NB: cutoff = nR * dR is redundant
   fgets(tmp, sizeof(tmp), potFile);
   sscanf(tmp, "%d %le %d %le %le", &nRho, &dRho, &nR, &dR, &cutoff);
   pot->cutoff = cutoff;

   // **** THIS CODE IS RESTRICTED TO ONE ELEMENT
   // Per-atom header 
   fgets(tmp, sizeof(tmp), potFile);
   int nAtomic;
   double mass, lat;
   char latticeType[8];
   sscanf(tmp, "%d %le %le %s", &nAtomic, &mass, &lat, latticeType);
   pot->atomicNo = nAtomic;
   pot->lat = lat;
   pot->mass = mass * amuToInternalMass;  // file has mass in AMU.
   strcpy(pot->latticeType, latticeType);
   
   // allocate read buffer
   int bufSize = MAX(nRho, nR);
   real_t* buf = (real_t*)comdMalloc(bufSize * sizeof(real_t));
   real_t x0 = 0.0;

   // Read embedding energy F(rhobar)
   for (int ii=0; ii<nRho; ++ii)
      fscanf(potFile, FMT1, buf+ii);
   pot->f = initInterpolationObject(nRho, x0, dRho, buf);

   // Read electron density rho(r)
   for (int ii=0; ii<nR; ++ii)
      fscanf(potFile, FMT1, buf+ii);
   pot->rho = initInterpolationObject(nR, x0, dR, buf);

   // Read phi(r)*r and convert to phi(r)
   for (int ii=0; ii<nR; ++ii)
      fscanf(potFile, FMT1, buf+ii);
   for (int ii=1; ii<nR; ++ii)
   {
      real_t r = x0 + ii*dR;
      buf[ii] /= r;
   }
   buf[0] = buf[1] + (buf[1] - buf[2]); // Linear interpolation to get phi[0].
   pot->phi = initInterpolationObject(nR, x0, dR, buf);

   comdFree(buf);

   // write to text file for comparison, currently commented out
/*    printPot(pot->f, "SetflDataF.txt"); */
/*    printPot(pot->rho, "SetflDataRho.txt"); */
/*    printPot(pot->phi, "SetflDataPhi.txt");  */
}

/// Reads potential data from a funcfl file and populates
/// corresponding members and InterpolationObjects in an EamPotential.
/// 
/// funcfl is a file format for tabulated potential functions used by
/// the original EAM code DYNAMO.  A funcfl file contains an EAM
/// potential for a single element.
/// 
/// The contents of a funcfl file are:
///
/// | Line Num | Description
/// | :------: | :----------
/// | 1        | comments
/// | 2        | elem amass latConstant latType
/// | 3        | nrho   drho   nr   dr    rcutoff
/// | 4        | embedding function values F(rhobar) starting at rhobar=0
/// |    ...   | (nrho values. Multiple values per line allowed.)
/// | x'       | electrostatic interation Z(r) starting at r=0
/// |    ...   | (nr values. Multiple values per line allowed.)
/// | y'       | electron density values rho(r) starting at r=0
/// |    ...   | (nr values. Multiple values per line allowed.)
///
/// Where:
///    -  elem          :   atomic number for this element
///    -  amass         :   atomic mass for this element in AMU
///    -  latConstant   :   lattice constant for this elemnent in Angstroms
///    -  lattticeType  :   lattice type for this element (e.g. FCC) 
///    -  nrho          :   number of values for the embedding function, F(rhobar)
///    -  drho          :   table spacing for rhobar
///    -  nr            :   number of values for Z(r) and rho(r)
///    -  dr            :   table spacing for r in Angstroms
///    -  rcutoff       :   potential cut-off distance in Angstroms
///
/// funcfl format stores the "electrostatic interation" Z(r).  This needs to
/// be converted to the pair potential phi(r).
/// using the formula 
/// \f[phi = Z(r) * Z(r) / r\f]
/// NB: phi is not defined for r = 0
///
/// Z(r) is in atomic units (i.e., sqrt[Hartree * bohr]) so it is
/// necesary to convert to eV.
///
/// F(rhobar) is in eV.
///
void eamReadFuncfl(EamPotential* pot, const char* dir, const char* potName)
{
   char tmp[4096];

#if defined(_WIN32) || defined(_WIN64)
   sprintf(tmp, "%s\\%s", dir, potName);
#else
   sprintf(tmp, "%s/%s", dir, potName);
#endif

   FILE* potFile = fopen(tmp, "r");
   if (potFile == NULL)
      fileNotFound("eamReadFuncfl", tmp);

   // line 1
   fgets(tmp, sizeof(tmp), potFile);
   char name[3];
   sscanf(tmp, "%s", name);
   strcpy(pot->name, name);

   // line 2
   int nAtomic;
   double mass, lat;
   char latticeType[8];
   fgets(tmp,sizeof(tmp),potFile);
   sscanf(tmp, "%d %le %le %s", &nAtomic, &mass, &lat, latticeType);
   pot->atomicNo = nAtomic;
   pot->lat = lat;
   pot->mass = mass*amuToInternalMass; // file has mass in AMU.
   strcpy(pot->latticeType, latticeType);

   // line 3
   int nRho, nR;
   double dRho, dR, cutoff;
   fgets(tmp,sizeof(tmp),potFile);
   sscanf(tmp, "%d %le %d %le %le", &nRho, &dRho, &nR, &dR, &cutoff);
   pot->cutoff = cutoff;
   real_t x0 = 0.0; // tables start at zero.

   // allocate read buffer
   int bufSize = MAX(nRho, nR);
   real_t* buf = (real_t*)comdMalloc(bufSize * sizeof(real_t));

   // read embedding energy
   for (int ii=0; ii<nRho; ++ii)
      fscanf(potFile, FMT1, buf+ii);
   pot->f = initInterpolationObject(nRho, x0, dRho, buf);

   // read Z(r) and convert to phi(r)
   for (int ii=0; ii<nR; ++ii)
      fscanf(potFile, FMT1, buf+ii);
   for (int ii=1; ii<nR; ++ii)
   {
      real_t r = x0 + ii*dR;
      buf[ii] *= buf[ii] / r;
      buf[ii] *= hartreeToEv * bohrToAngs; // convert to eV
   }
   buf[0] = buf[1] + (buf[1] - buf[2]); // linear interpolation to get phi[0].
   pot->phi = initInterpolationObject(nR, x0, dR, buf);

   // read electron density rho
   for (int ii=0; ii<nR; ++ii)
      fscanf(potFile, FMT1, buf+ii);
   pot->rho = initInterpolationObject(nR, x0, dR, buf);

   comdFree(buf);
   
/*    printPot(pot->f,   "funcflDataF.txt"); */
/*    printPot(pot->rho, "funcflDataRho.txt"); */
/*    printPot(pot->phi, "funcflDataPhi.txt"); */
}

void fileNotFound(const char* callSite, const char* filename)
{
   fprintf(screenOut,
           "%s: Can't open file %s.  Fatal Error.\n", callSite, filename);
   exit(-1);
}

void notAlloyReady(const char* callSite)
{
   fprintf(screenOut,
          "%s: CoMD 1.1 does not support alloys and cannot\n"
           "   read setfl files with multiple species.  Fatal Error.\n", callSite);
   exit(-1);
}

void typeNotSupported(const char* callSite, const char* type)
{
   fprintf(screenOut,
          "%s: Potential type %s not supported. Fatal Error.\n", callSite, type);
   exit(-1);
}

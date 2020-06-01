/// \file
/// Leapfrog time integrator

#include "timestep.h"

#include "CoMDTypes.h"
#include "linkCells.h"
#include "parallel.h"
#include "performanceTimers.h"

// computes local potential and kinetic energies
EXTERN_C void computeEnergy(SimFlat *sim, real_t *eLocal);	

EXTERN_C void advanceVelocity(SimGpu sim, real_t dt);
EXTERN_C void advancePosition(SimGpu sim, real_t dt);

EXTERN_C void updateLinkCells(SimFlat *sim);
EXTERN_C void sortAtomsGpu(SimFlat *sim, cudaStream_t stream);

EXTERN_C void updateNeighborsGpuAsync(SimGpu sim, int *temp, int num_cells, int *cell_list, cudaStream_t stream);

EXTERN_C void eamForce1GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream);
EXTERN_C void eamForce2GpuAsync(SimGpu sim, AtomListGpu list, cudaStream_t stream);

/// Advance the simulation time to t+dt using a leap frog method
/// (equivalent to velocity verlet).
///
/// Forces must be computed before calling the integrator the first time.
///
///  - Advance velocities half time step using forces
///  - Advance positions full time step using velocities
///  - Update link cells and exchange remote particles
///  - Compute forces
///  - Update velocities half time step using forces
///
/// This leaves positions, velocities, and forces at t+dt, with the
/// forces ready to perform the half step velocity update at the top of
/// the next call.
///
/// After nSteps the kinetic energy is computed for diagnostic output.
double timestep(SimFlat* s, int nSteps, real_t dt)
{
   for (int ii=0; ii<nSteps; ++ii)
   {
      startTimer(velocityTimer);
      advanceVelocity(s->gpu, 0.5*dt); 
      stopTimer(velocityTimer);

      startTimer(positionTimer);
      advancePosition(s->gpu, dt);
      stopTimer(positionTimer);

      startTimer(redistributeTimer);
      redistributeAtoms(s);
      stopTimer(redistributeTimer);

      startTimer(computeForceTimer);
      computeForce(s);
      stopTimer(computeForceTimer);

      startTimer(velocityTimer);
      advanceVelocity(s->gpu, 0.5*dt); 
      stopTimer(velocityTimer);
   }

   kineticEnergyGpu(s);

   return s->ePotential;
}

void computeForce(SimFlat* s)
{
   s->pot->force(s);
}

/// Calculates total kinetic and potential energy across all tasks.  The
/// local potential energy is a by-product of the force routine.
void kineticEnergy(SimFlat* s)
{
   real_t eLocal[2];
   eLocal[0] = s->ePotential;
   eLocal[1] = 0;
   for (int iBox=0; iBox<s->boxes->nLocalBoxes; iBox++)
   {
      for (int iOff=MAXATOMS*iBox,ii=0; ii<s->boxes->nAtoms[iBox]; ii++,iOff++)
      {
         int iSpecies = s->atoms->iSpecies[iOff];
         real_t invMass = 0.5/s->species[iSpecies].mass;
         eLocal[1] += ( s->atoms->p[iOff][0] * s->atoms->p[iOff][0] +
         s->atoms->p[iOff][1] * s->atoms->p[iOff][1] +
         s->atoms->p[iOff][2] * s->atoms->p[iOff][2] )*invMass;
      }
   }

   real_t eSum[2];
   startTimer(commReduceTimer);
   addRealParallel(eLocal, eSum, 2);
   stopTimer(commReduceTimer);

   s->ePotential = eSum[0];
   s->eKinetic = eSum[1];
}

/// Calculates total kinetic and potential energy across all tasks.  The
/// local potential energy is a by-product of the force routine.
void kineticEnergyGpu(SimFlat* s)
{
   real_t eLocal[2];

   computeEnergy(s, eLocal);

   real_t eSum[2];
   startTimer(commReduceTimer);
   addRealParallel(eLocal, eSum, 2);
   stopTimer(commReduceTimer);

   s->ePotential = eSum[0];
   s->eKinetic = eSum[1];
}

/// \details
/// This function provides one-stop shopping for the sequence of events
/// that must occur for a proper exchange of halo atoms after the atom
/// positions have been updated by the integrator.
///
/// - updateLinkCells: Since atoms have moved, some may be in the wrong
///   link cells.
/// - haloExchange (atom version): Sends atom data to remote tasks. 
/// - sort: Sort the atoms.
///
/// \see updateLinkCells
/// \see initAtomHaloExchange
/// \see sortAtomsInCell
void redistributeAtoms(SimFlat* sim)
{ 
   updateLinkCells(sim);

   // cell lists are updated 
   // now we can launch force computations on the interior
   if (sim->gpuAsync) {
     // only update neighbors list when method != thread_atom
     if (sim->gpu.method > 1) 
       updateNeighborsGpuAsync(sim->gpu, sim->flags, sim->gpu.n_local_cells - sim->n_boundary_cells, sim->interior_cells, sim->interior_stream);

     eamForce1GpuAsync(sim->gpu, sim->gpu.i_list, sim->interior_stream);
     eamForce2GpuAsync(sim->gpu, sim->gpu.i_list, sim->interior_stream);
   }

   // exchange is only for boundaries
   startTimer(atomHaloTimer);
   haloExchange(sim->atomExchange, sim);
   stopTimer(atomHaloTimer);

   // sort only boundary cells
   sortAtomsGpu(sim, sim->boundary_stream);
}

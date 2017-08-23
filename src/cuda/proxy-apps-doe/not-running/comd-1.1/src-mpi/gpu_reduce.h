
__global__ void ReduceEnergy(SimGpu sim, real_t *e_pot, real_t *e_kin)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  int iAtom = sim.a_list.atoms[tid];
  int iBox = sim.a_list.cells[tid];

  int iOff = iBox * N_MAX_ATOMS + iAtom;

  real_t ep = 0;
  real_t ek = 0; 
  if (tid < sim.n_local_atoms) {
    int iSpecies = sim.species_ids[iOff];
    real_t invMass = 0.5/sim.species_mass[iSpecies];
    ep = sim.e[iOff]; 
    ek = (sim.p.x[iOff] * sim.p.x[iOff] + sim.p.y[iOff] * sim.p.y[iOff] + sim.p.z[iOff] * sim.p.z[iOff]) * invMass;
  }
  
  // reduce in smem
  __shared__ real_t sp[THREAD_ATOM_CTA];
  __shared__ real_t sk[THREAD_ATOM_CTA];
  sp[threadIdx.x] = ep;
  sk[threadIdx.x] = ek;
  __syncthreads();
  for (int i = THREAD_ATOM_CTA / 2; i >= WARP_SIZE; i /= 2) {
    if (threadIdx.x < i) {
      sp[threadIdx.x] += sp[threadIdx.x + i];
      sk[threadIdx.x] += sk[threadIdx.x + i];
    }
    __syncthreads();
  }
  
  // reduce in warp
  if (threadIdx.x < 32) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 300
    ep = sp[threadIdx.x];
    ek = sk[threadIdx.x];
    for (int i = WARP_SIZE / 2; i > 0; i /= 2) {
      ep += __shfl_xor(ep, i);
      ek += __shfl_xor(ek, i);
    }
#else

#if WARP_SIZE > 16
    if (threadIdx.x < 16) sp[threadIdx.x] += sp[threadIdx.x+16];
    _Z_intrinsic_pseudo_syncwarp();
#endif
#if WARP_SIZE > 8
    if (threadIdx.x < 8) sp[threadIdx.x] += sp[threadIdx.x+8];
    _Z_intrinsic_pseudo_syncwarp();
#endif
#if WARP_SIZE > 4
    if (threadIdx.x < 4) sp[threadIdx.x] += sp[threadIdx.x+4];
    _Z_intrinsic_pseudo_syncwarp();
#endif

    if (threadIdx.x < 2) sp[threadIdx.x] += sp[threadIdx.x+2];
    _Z_intrinsic_pseudo_syncwarp();
    if (threadIdx.x < 1) sp[threadIdx.x] += sp[threadIdx.x+1];
    _Z_intrinsic_pseudo_syncwarp();

#if WARP_SIZE > 16
    if (threadIdx.x < 16) sk[threadIdx.x] += sk[threadIdx.x+16];
    _Z_intrinsic_pseudo_syncwarp();
#endif
#if WARP_SIZE > 8
    if (threadIdx.x < 8) sk[threadIdx.x] += sk[threadIdx.x+8];
    _Z_intrinsic_pseudo_syncwarp();
#endif
#if WARP_SIZE > 4
    if (threadIdx.x < 4) sk[threadIdx.x] += sk[threadIdx.x+4];
    _Z_intrinsic_pseudo_syncwarp();
#endif

    if (threadIdx.x < 2) sk[threadIdx.x] += sk[threadIdx.x+2];
    _Z_intrinsic_pseudo_syncwarp();
    if (threadIdx.x < 1) sk[threadIdx.x] += sk[threadIdx.x+1];
    _Z_intrinsic_pseudo_syncwarp();

    if (threadIdx.x == 0) {
      ep = sp[threadIdx.x];
      ek = sk[threadIdx.x];
    }
#endif
  }

  // one thread adds to gmem
  if (threadIdx.x == 0) {
    atomicAdd(e_pot, ep);
    atomicAdd(e_kin, ek);
  }
}

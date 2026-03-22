# H100 Benchmark Suite

14 modern GPU workloads from H100 profiling and analysis.

## Applications

### cuFFT (2 apps) - FFT operations
- `cufft_3d_c2c_scalable` - 3D Complex-to-Complex FFT
- `cufft_lto_r2c_c2r_scalable` - Real↔Complex FFT with LTO callbacks

### cuSolver (2 apps) - Linear algebra
- `cusolver_ormqr_scalable` - QR factorization
- `cusolver_Xgetrf_scalable` - LU factorization

### Image Processing (3 apps)
- `dwtHaar1D` - Haar wavelet transform
- `recursiveGaussian` - Recursive Gaussian filter
- `FDTD3d` - Finite-Difference Time-Domain 3D simulation

### Graph Algorithms (2 apps)
- `bfs_standalone` - Breadth-First Search (requires cuGraph submodule)
- `mst_standalone` - Minimum Spanning Tree (requires cuGraph submodule)

### Physics Simulation (3 apps)
- `newton_diffsim_ball` - Differential simulation (requires Newton submodule)
- `newton_robot_cartpole` - Robotics simulation (requires Newton submodule)
- `newton_mpm_granular` - Material Point Method simulation (requires Newton submodule)

### Computer Vision (3 apps)
- `vpi_background_subtractor` - Background subtraction (requires VPI 4.0)
- `vpi_orb_feature_detector` - ORB feature detection (requires VPI 4.0)
- `vpi_stereo_disparity` - Stereo disparity calculation (requires VPI 4.0)

## Dependencies

- **CUDA 11.0+** - Required (provides cuFFT, cuSolver, cuBLAS libraries)
- **cuGraph** - Git submodule (auto-initialized for graph apps)
- **Newton** - Git submodule (auto-initialized for physics apps)
- **VPI 4.0** - install from https://developer.nvidia.com/embedded/vpi

## Build

```bash
# From repository root
source src/setup_environment

# Generate data files (standard workflow)
make data

# Build all H100 apps
make -C src H100

# Or build everything with:
make all -i -j -C src
```

Binaries are output to `bin/<cuda-version>/release/H100-*`

Newton apps are copied to `bin/<cuda-version>/release/newton/newton_*`

## Running

```bash
# cuFFT apps
bin/*/release/H100-cufft_3d_c2c small
bin/*/release/H100-cufft_lto_r2c_c2r medium

# cuSolver apps
bin/*/release/H100-cusolver_ormqr large
bin/*/release/H100-cusolver_Xgetrf medium

# Image apps
bin/*/release/H100-dwtHaar1D
bin/*/release/H100-recursiveGaussian
bin/*/release/H100-FDTD3d

# Graph apps (with generated data)
bin/*/release/H100-bfs data_dirs/cuda/H100/graph/karate.mtx
bin/*/release/H100-mst data_dirs/cuda/H100/graph/netscience.mtx

# Newton apps
bin/*/release/newton/newton_diffsim_ball
bin/*/release/newton/newton_robot_cartpole
bin/*/release/newton/newton_mpm_granular

# VPI apps (if VPI installed)
bin/*/release/vpi_background_subtractor cuda <video-file> <frames>
```

## GPU Support

- Requires compute capability 7.5+ (Turing, Ampere, Hopper)
- Tested on: V100 (sm_70), A100 (sm_80), H100 (sm_90)

## Notes

- Simple apps (cuFFT, cuSolver, image) build on any system with CUDA 11+
- Graph apps require cuGraph submodule (automatically handled by build system)
- Newton apps create Python virtual environment on first run
- VPI apps are optional and only build if VPI library is installed

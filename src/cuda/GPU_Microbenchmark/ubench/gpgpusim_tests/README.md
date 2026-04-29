# GPGPU-Sim Standalone Tests

This directory collects standalone CUDA/PTX tests that were originally kept
with GPGPU-Sim feature branches. They are placed under `GPU_Microbenchmark` so
the app-collection build can compile them with the same flow as the other
microbenchmarks.

The tests here intentionally avoid linking against GPGPU-Sim internals. C++
unit tests that need simulator headers or libraries should stay in the
GPGPU-Sim source tree.

## Test Groups

- `fp_lowp_cvt`: runtime low-precision conversion kernels and PTX cases used
  for `ptxas` cross-checking.
- `tma_tensormap`: standalone TMA tensor-map benchmark plus expected output
  fixtures from the simulator branch.

# FP Low-Precision Conversion Tests

This folder contains standalone CUDA/PTX low-precision conversion tests from
the GPGPU-Sim `fp-work` branch.

## Build

```sh
make
```

The default build creates these binaries under `GPU_Microbenchmark/bin`:

- `lowp_cvt_runtime_hw_sim_diff`
- `lowp_cvt_runtime_hw_sim_diff_rs`

Default architectures are `LOWP_ARCH=sm_120a` and
`LOWP_RS_ARCH=sm_100a`. Override them on the make command line if needed.

## PTX Cross-Check

```sh
make ptxas-crosscheck
```

This target runs the imported PTX accept/reject cases through `ptxas`.

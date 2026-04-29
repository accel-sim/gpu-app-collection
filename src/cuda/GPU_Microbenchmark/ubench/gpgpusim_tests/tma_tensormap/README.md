# TMA Tensor-Map Test

This folder contains the standalone TMA tensor-map benchmark from the
GPGPU-Sim `tma` branch. Expected fixture outputs are kept under `expected/`.

## Build

```sh
make
```

The default build creates `gpgpusim_tma_tensor_benchmark` under
`GPU_Microbenchmark/bin`.

## Fixture Check

```sh
make run-fixtures
```

This target runs the imported fixture cases and compares generated output
against `expected/`.

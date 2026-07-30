# `%cluster_ctarank` microbenchmark

This benchmark launches multiple one-dimensional thread-block clusters and
records the PTX `%cluster_ctarank` special register once per CTA. For a cluster
containing `N` CTAs, the expected ranks are `0` through `N - 1`; the sequence
starts again at zero in the next cluster. It also reads `%ctaid.x` directly and
checks that it equals CUDA's `blockIdx.x`, the CTA's grid-wide x coordinate.

Build for an RTX 5070/5070 Ti (`sm_120`) and run the default case (four clusters,
four CTAs per cluster):

```sh
make
../../../bin/cluster_ctarank
```

Choose another number of clusters and CTAs per cluster at runtime:

```sh
../../../bin/cluster_ctarank 3 8
```

The grid always contains `clusters * ctas-per-cluster` CTAs. The second argument
is limited to the portable maximum of eight CTAs per cluster.

## Three-dimensional test

`cluster_ctarank_3d` uses a `2 x 2 x 2` cluster shape and a `4 x 4 x 4` grid,
giving eight clusters with eight CTAs each. It reads `%ctaid.x`, `%ctaid.y`, and
`%ctaid.z` directly and compares them with the corresponding `blockIdx`
components. It also checks the flattened cluster-local rank:

```text
rank = local_x + 2 * (local_y + 2 * local_z)
```

Run it with:


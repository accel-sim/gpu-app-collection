### The Benchmarks

The initial iteration of this repository contains a set of functional tests based on Rodinia 2.0 created by Andrew Boktor from UBC.
Over time, we will add workloads to this tree for other updated benchmarks suites.
It is also a long-term goal that papers published using GPGPU-Sim can submit their workloads here as a centralized place where people looking to reproduce work can gather them.
The SDK is conspicuously absent due to the lengthy license agreement associated with the SDK code. It might be alright to post it here, but we don't have time to make sure.

#### ./benchmarks/data_dirs

The repo itself has no data (since git is bad with big files). git-lfs is one option that could be explored, but since the public, free version of github limits the
capacity and bandwidth using git-lfs, the data is simply retrieved via wget form a tarball hosted on Purdue's servers.

#### ./benchmarks/src/

This is where all the code for the apps go.
A top-level makefile  is here for allowing all of these to be built easily. In the initial commit, only rodinia-2.0-ft apps are included, but more will be added later.
It should also be noted that the common.mk file from the CUDA 4.2 SDK is included here, since the rodinia benchmarks rely on it.

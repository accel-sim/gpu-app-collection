### The Benchmarks

This repository contains a collection of the most common applications run in simulation. We continuously update this repo over time as different apps become more important.
One of the primary goals of this repo is to provide a centralized place for apps that continue to build with different versions of CUDA.
Many of the open-source repos for these applications, do not update the build infrastructure such that they build with modern CUDA, while still
proving a way to biuld them with older versions (which may be usful for some apps).
The apps in this repo can be built with CUDA 4.2 through 11.0 by simply doing:
```
# Make sure CUDA_INSTALL_PATH is set.
source ./src/setup_environment
make all -i -j -C ./src
make data # pulls all the data files the apps need to run and puts them in a centralized location
```

Some notes:
- Some apps have additional dependencies beyond what Accel-Sim requires. This is why we recommend building with the "-i" flag to see what you can get to build on your current system. Our internal regressions verify that all the apps do build, but covering all their dependencies can be difficult.
To see all the apps that built successfully, run:
```
ls ./bin/<cuda-vers>/release
```
- We did not wite many of these application suites - but we have tried to mainain their original structure
and copyright information. If you use the apps in the suite with this infrastructure be sure to cite both
the Accel-Sim paper that introduces the build infrastructure, and indicate which version (i.e. label or commit#)
of this repo you used in your paper:
```
Mahmoud Khairy, Zhensheng Shen, Tor M. Aamodt, Timothy G. Rogers,
Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling,
in 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA)
```
- This suite works easiest when launching work with Accel-Sim Framework: https://github.com/accel-sim/accel-sim-framework

#### ./benchmarks/data_dirs

The repo itself has no data (since git is bad with big files).
git-lfs is one option that could be explored, but since the public, free version of github limits the
capacity and bandwidth using git-lfs, the data is simply retrieved via wget form a tarball hosted on our University servers.
```
./get_data.sh
```
will grab this data, as well as:
```
make data
```

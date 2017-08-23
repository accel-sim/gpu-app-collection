# Dragon_li

`Dragon_li` is a benchmark suite of irregular applications utilizing the CUDA Dynamic 
Parallelism (CDP) features on NVIDIA's GPU with GK110 architecture or above (sm_35 or above).
The purpose of `Dragon_li` is to facilitate investigation and studies on irregular applications
that feature fine-grained dynamic parallelism on modern GPU architectures.

The current version of `Dragon_li` is only a preliminary release. The authors are continuously
adding more features to the exsiting benchmarks, new benchmarks and documents. Please check
out the [Dragon_li project site](http://gpuocelot.gatech.edu/dragon_li/) for more information. 

## Check out

To check out the source code, first run:

`git clone https://github.com/gtcasl/dragon_li.git`

`Dragon_li` has external dependency on `Hydrainze` 
([https://github.com/gtcasl/hydrazine](https://github.com/gtcasl/hydrazine)) which is 
defined as a submodule. The `Hydrazine` submodule can be initialized and feched under the
top dir of `Dragon_li`:

`git submodule init; git submodule update`

## Install
- `Scons`([http://www.scons.org/](http://www.scons.org/)) is required to build `Dragon_li`. On Ubuntu,
`Scons` can be installed by running:

`apt-get install scons`.

- CUDA toolkit 6.5 is required to build `Dragon_li`. CUDA toolkit will be located under `/usr/local/cuda/`.
You may modifly `sconscript` with your own CUDA toolkit installation path. CUDA toolkit with a version
number higher than 6.5 may also work but has not been tested.

- Under the top dir of `Dragon_li`, run:

`scons`

- To build the benchmark with CDP, run:

`scons cdp=1 no_debug=1`

- You may get a list of building options by running:

`scons -h`

## Run

The generated executables are installed uner `dragon_li_top_dir/bin/. You may run each individual 
executable with "`-h`" to look up available options. For example:

`testBfs -h`

## Test with GPGPU-Sim

- Use the config files in `gpgpusim_config` under this branch.
- Download the graphs used by the benchmarks from [http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/citationCiteseer.graph.bz2](http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/citationCiteseer.graph.bz2) and [http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/coPapersDBLP.graph.bz2](http://www.cc.gatech.edu/dimacs10/archive/data/coauthor/coPapersDBLP.graph.bz2). Unzip and save the graphs under `top_dir/graphs`.
- The followings are eight different benchmark tests for GPGPUSim. "Verify correct" will be shown in the output to indicate the benchmark runs correctly.

`./bin/testBfs -g graphs/sample_cdp.gr -e -v --cdp`

`./bin/testBfs -g graphs/citationCiteseer.graph -f metis -e -v --cdp`

`./bin/testBfs -g graphs/coPapersDBLP.graph -f metis -e -v --cdp --sf 1.5`

`./bin/testAmr -v -e --cdp -r 20`

`./bin/testSssp -g graphs/sample_cdp.gr -e -v --cdp`

`./bin/testSssp -g graphs/citationCiteseer.graph -f metis -e -v --cdp`

`./bin/testSssp -g graphs/coPapersDBLP.graph -f metis -e -v --cdp`

`./bin/testJoin -v -e --cdp -l 204800 -r 204800`


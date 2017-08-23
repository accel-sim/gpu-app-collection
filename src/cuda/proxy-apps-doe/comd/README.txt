CoMD provides cmd line options to control grid size and implementation to run.

This will run single EAM step for 20x20x20 grid (default grid size):
./CoMDCUDA -p ag -e -n 0

If you want to change grid size, use -x -y -z options. 
For example the following cmd will run single EAM step for 10x10x10 grid:
./CoMDCUDA -p ag -e -n 0 -x 10 -y 10 -z 10 

If you want to switch to a different implementation, specify -m and name of the method. 
The following will run 10x10x10 and use cta per box approach:
./CoMDCUDA -p ag -e -n 0 -x 10 -y 10 -z 10 -m cta_box

Available implementations:
* thread_atom_warp_sync (default)
* thread_atom
* cta_box
* cta_box_agg (currently produces incorrect results on ECX)

If you want to use something ECX specific in your code you need to wrap this into ifdef on ECX_TARGET

Sample command lines to submit jobs to LSF:

* 1 TPC, using thread_atom_warp_sync, 20x20x20:
qsub -P einstein -q o_cpu_16G_4H -n 4 -N -o perf_tests/thread_atom_warp_sync_20.log ./ecx_run_1tpc_job.sh -x 20 -y 20 -z 20 -m thread_atom_warp_sync

* 1 SM, using thread_atom, 10x10x10:
qsub -P einstein -q o_cpu_16G_1H -n 4 -N -o perf_tests/thread_atom_10.log ./ecx_run_1sm_job.sh -x 10 -y 10 -z 10 -m thread_atom

View status of your jobs:
bjobs

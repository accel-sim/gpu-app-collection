ECX_COMP_OPT=-arch=sm_20 -O3 -Xptxas -dlcm=ca -Xptxas -maxrregcount=63 -Xptxas -v

all:
	nvcc $(ECX_COMP_OPT)                                                             sweep.cu -o sweep_full
	nvcc $(ECX_COMP_OPT) -DREMOVE_REDUCE                                             sweep.cu -o sweep_remove_reduce
	nvcc $(ECX_COMP_OPT) -DREMOVE_REDUCE -DREMOVE_UNBALANCE                          sweep.cu -o sweep_remove_reduce_unbal
	nvcc $(ECX_COMP_OPT) -DREMOVE_REDUCE -DREMOVE_UNBALANCE -DREMOVE_SYNC_AFTER_DIAG sweep.cu -o sweep_remove_reduce_unbal_sync
	nvcc $(ECX_COMP_OPT) -DREMOVE_REDUCE                    -DREMOVE_SYNC_AFTER_DIAG sweep.cu -o sweep_remove_reduce_sync
	nvcc $(ECX_COMP_OPT)                                    -DREMOVE_SYNC_AFTER_DIAG sweep.cu -o sweep_remove_sync

#make for real K40 or K20
real:
	nvcc -O3 -Xptxas -dlcm=ca -Xptxas -maxrregcount=63 -Xptxas -v sweep.cu -o sweep


clean:
	rm sweep_* -f

#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets sssp.cu -lcudart -o sssp

prefetch:
	nvcc -g -DPREF -Wno-deprecated-gpu-targets sssp.cu -lcudart -o sssp

clean:
	rm -rf sssp gpgpusim_power_report* _cuobjdump_* *.txt

#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets atax.cu -lcudart -o atax

clean:
	rm -rf atax gpgpusim_power_report* _cuobjdump_* *.txt

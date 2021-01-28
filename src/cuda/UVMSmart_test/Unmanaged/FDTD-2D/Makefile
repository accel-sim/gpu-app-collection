#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets fdtd2d.cu -lcudart -o main

clean:
	rm -rf main gpgpusim_power_report* _cuobjdump_* *.txt

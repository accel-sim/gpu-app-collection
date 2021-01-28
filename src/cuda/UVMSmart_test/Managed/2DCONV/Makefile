#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets 2DConvolution.cu -lcudart -o 2dconv

clean:
	rm -rf 2dconv gpgpusim_power_report* _cuobjdump_* *.txt

#!/usr/bin/make

default:
	nvcc -g -Wno-deprecated-gpu-targets stream_triad.cu -lcudart -o stream_triad

clean:
	rm -rf stream_triad gpgpusim_power_report* _cuobjdump_* *.txt

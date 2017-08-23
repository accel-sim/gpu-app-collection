#!/bin/bash

for bench in bc color fw mis pagerank sssp
do
	echo $bench
	pushd . >& /dev/null
	cd $bench
	make clean; make clean-gem5-fusion
	popd >& /dev/null
done

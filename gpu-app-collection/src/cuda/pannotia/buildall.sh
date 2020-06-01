#!/bin/bash

extraparams=""
buildparams=""
if [ ! -z "$1" ]
then
	extraparams="$1"
	buildparams="gem5-fusion"
fi

function savebin {
	if [ "$2" == "" ]
	then
		backupbench="$1"
	else
		backupbench="gem5_fusion_$1"
	fi
	if [ -e "$backupbench" ]
	then
		echo "Saving bin $backupbench"
		mv $backupbench $backupbench.bak
	fi
}

function restorebin {
	if [ "$2" == "" ]
	then
		backupbench="$1"
	else
		backupbench="gem5_fusion_$1"
	fi
	if [ -e "$backupbench.bak" ]
	then
		echo "Restoring bin $backupbench"
		mv $backupbench.bak $backupbench
	fi
}

for bench in bc color fw mis pagerank sssp
do
	echo $bench
	pushd . >& /dev/null
	cd $bench
	make clean; make clean-gem5-fusion
	make $buildparams $extraparams
	if [ "$bench" == "color" ]
	then
		savebin "color_max" $buildparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=MAXMIN $extraparams
		restorebin "color_max" $buildparams
	elif [ "$bench" == "fw" ]
	then
		savebin "fw" $buildparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=BLOCK $extraparams
		restorebin "fw" $buildparams
	elif [ "$bench" == "pagerank" ]
	then
		savebin "pagerank" $buildparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=SPMV $extraparams
		restorebin "pagerank" $buildparams
	elif [ "$bench" == "sssp" ]
	then
		savebin "sssp" $buildparams
		make clean; make clean-gem5-fusion
		make $buildparams VARIANT=ELL $extraparams
		restorebin "sssp" $buildparams
	fi
	popd >& /dev/null
done

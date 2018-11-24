#!/bin/bash
export REPLACE_ITER_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

if [ $# = 1 ]; then
	if [ "$1" = "HW" ]; then
		platform="HW"
	elif [ "$1" = "SM" ]; then
		platform="SM"
	else
		echo "Usage: replace_iterations.sh <HW/SM>"
		exit 1
	fi
else
	echo "Usage: replace_iterations.sh <HW/SM>"
	exit 1
fi

ROOT_DIR=$REPLACE_ITER_DIR
directories=`grep -E '^[^#].*' "$REPLACE_ITER_DIR/directory.list"`
for bench_group in $directories
do
	cd "$ROOT_DIR/$bench_group"
	benchmarks=`ls -d */`
	for bench_dir in $benchmarks
	do
		cd $bench_dir
		iter_file="iterations.$platform"
		if [ -e $iter_file ] && [ -f $iter_file ]; then
			iterations=`sed '1!d' $iter_file`
			cuda_file=`ls | grep -E '.cu$'`		
			cp -f $cuda_file $cuda_file".backup"
			sed -E -i "s/REPLACE_ITERATIONS/$iterations/g" $cuda_file
			flags_file="flags.HW"
			if [ -e $flags_file ] && [ -f $flags_file ]; then
				flags=`sed '1!d' $flags_file`
				cp "Makefile" "Makefile_bu"
				sed -i "s/REPLACE_FLAGS/$flags/g" "Makefile"
			fi
		else
			files_needing_replacement=`grep 'REPLACE_ITERATIONS' ./*.cu`
			if ! [ "$files_needing_replacement" = '' ]; then
				echo "Error: iterations file missing" >&2;
				cd $THIS_DIR
				./restore_backups.sh
				exit 1
			fi
		fi
		cd ..
	done
done

#cd $ROOT_DIR
#make -j8 -k power 
#cd $THIS_DIR
#./restore_backups.sh "$1" "$2"

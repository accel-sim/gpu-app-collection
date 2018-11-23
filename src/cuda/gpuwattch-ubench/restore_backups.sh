#!/bin/bash
export REPLACE_ITER_DIR="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"

if [ $# = 2 ]; then
	if [ "$1" = "HW" ]; then
		platform="HW"
	elif [ "$1" = "SM" ]; then
		platform="SM"
	else
		echo "Usage: restore_backups.sh <HW/SM> <POWER/PERF>"
		exit 1
	fi
else
	echo "Usage: restore_backups.sh <HW/SM> <POWER/PERF>"
	exit 1
fi

ROOT_DIR="$REPLACE_ITER_DIR"
directories=`grep -E '^[^#].*' "$REPLACE_ITER_DIR/directory.list"`
for bench_group in $directories
do
	cd "$ROOT_DIR/$bench_group"
	benchmarks=`ls -d */`
	for bench_dir in $benchmarks
	do
		cd $bench_dir
		backup_file=`ls | grep '.backup'`
		cuda_file=`echo $backup_file | sed 's/\.backup//g'`
		rm -f $cuda_file
		echo $cuda_file
		echo $backup_file
		mv $backup_file $cuda_file
		makefile_bu="Makefile_bu"
		if [ -f $makefile_bu ]; then
			rm -f "Makefile"
			mv $makefile_bu "Makefile"
		fi
		cd ..
	done
done

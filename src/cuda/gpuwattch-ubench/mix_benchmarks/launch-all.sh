#!/bin/bash

ASSNHOME=`pwd`
TOP_DIR=../../..
#echo ASSNHOME=$ASSNHOME
#echo TOP_DIR=$TOP_DIR
GPGPUSIM_CONFIG_FILES="$ASSNHOME/$TOP_DIR/simulation/scripts"
MCPAT_CONFIG_FILES="$ASSNHOME/$TOP_DIR/mcpat"
GPGPUSIM_DISTRIBUTION="$ASSNHOME/$TOP_DIR/distribution/"

BENCH_BIN_DIR="$ASSNHOME/$TOP_DIR/simulation/bin/release"

#GPGPUSIM_CONFIG="gpgpusim.config"
#INTER_CONFIG="icnt_config_fermi_islip.txt"

for bench in `ls .`
do
	if [ -d $bench ]; then	
		cd $ASSNHOME/$bench
		
		# Remove files
		rm -f _ptx*;
		rm -f gpgpusim_visu*;
		rm -f coeff_cycle.rpt;
		rm -f power_cycle.rpt;
		rm -f metric_cycle.rpt;
		rm -f max_power_file.rpt;
		rm -f avgmetric_cycle.rpt;
		rm -f avg_power_file.rpt;

		# Remove previous Torque run files... Comment this out if you want to keep the old run history
		rm -f $bench.o*;
		rm -f $bench.e*;

		# Copy over files
		if [ -f $GPGPUSIM_CONFIG_FILES/fermi.config ]; then
			cp -f $GPGPUSIM_CONFIG_FILES/fermi.config ./gpgpusim.config
		else
			echo "Please put fermi.config in simulation/scripts directory"
			exit
		fi
		if [ -f $GPGPUSIM_CONFIG_FILES/icnt_config_fermi_islip.txt  ]; then
			cp -f $GPGPUSIM_CONFIG_FILES/icnt_config_fermi_islip.txt ./
		else
			echo "Please put icnt_config_fermi_islip.txt in simulation/scripts directory"
			exit
		fi

		if [ -f $GPGPUSIM_CONFIG_FILES/fermi.xml ]; then	
			cp -f $MCPAT_CONFIG_FILES/fermi.xml ./
		else
			echo "Please put fermi.xml in simulation/scripts directory"
			exit
		fi

		cp "$ASSNHOME/torque.sh" ./ 
		sed -i "s#_BENCHMARK_BIN_DIR_#$BENCH_BIN_DIR#g" torque.sh
		sed -i "s#_BENCHMARK_NAME_#$bench#g" torque.sh
		sed -i "s#_GPGPUSIM_DISTRIBUTION_REPLACE_#$GPGPUSIM_DISTRIBUTION#g" torque.sh
		sed -i "s#_RUNDIR_REPLACE_#`pwd`#g" torque.sh

		# Launch the benchmark
		qsub -N "$bench" torque.sh

#		How to run only a subset of benchmarks 
#		if [ "$bench" == "add_mem_1" ]; then	
#			qsub -N "$bench" torque.sh
#			echo "$bench is running now"
#			
#		fi
		
		cd $ASSNHOME
	fi
	




done

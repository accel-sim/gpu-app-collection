#!/bin/bash
for i in `ls .`
do
	if [ -d $i ]; then
		if [ -f $i/avg_power_file.rpt ]; then
		let count=0	
		cat $i/avg_power_file.rpt | while read line
		do 
			if ((count!=0)); then	
		        echo -n $i;
			echo -n ',';
			echo $line;
		       fi
		       ((count++));
		done
			#echo $i;
			#awk '$0' $i/avg_power_file.rpt;
		fi
	fi
done

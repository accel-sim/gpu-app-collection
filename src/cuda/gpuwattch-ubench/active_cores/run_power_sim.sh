echo "Running all benchnarks..."

if [ ! -n "$POWER_BIN_DIR" ]; then
	echo "Please set env var "POWER_BIN_DIR" to directory containing binaries for the power benchmarks"
	exit 1
fi

for i in `ls .`
do
        if [ -d $i ]; then
		cd $i;
		rm -f gpgpusim_visu*;
		rm -f coeff_cycle.rpt;
		rm -f power_cycle.rpt;
		rm -f metric_cycle.rpt;
		rm -f max_power_file.rpt;
		rm -f avgmetric_cycle.rpt;
		rm -f avg_power_file.rpt;
		rm -f _ptx*;
		rm -f *.xml;
		cp -f ../../../../mcpat/fermi.xml ./mcpat.xml
		cp -f ../../../scripts/icnt_config_fermi_islip.txt ./
		cp -f ../../../scripts/fermi.config ./gpgpusim.config
		$POWER_BIN_DIR/$i> $i.log 2>&1 &
		echo -n "finshed "
		echo $i;
		cd ..
	fi
done

echo "Finished running all benchmarks..."

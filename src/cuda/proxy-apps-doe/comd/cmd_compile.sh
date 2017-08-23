if [ ! -z "$2" ]
then
	EMC_ARGS=''

	count=0
	for argument in "$@"
	do
		if [ $count -ge "1" ]
		then
			EMC_ARGS="$EMC_ARGS $argument"
		fi
		count=$((count+1))
	done
else
	EMC_ARGS='-v -k -f -fpermissive'
fi


if [ ! -z "$1" ]
then
	YAML_PATH=$1
else
	YAML_PATH=../../../../emc/machines/einstein_mm_clustered_scalar.yaml
fi

# voodoo hack: this should not be required...doing it anyway.
./clean.sh
make clean
make ECX_FLAGS=-DECX_TARGET

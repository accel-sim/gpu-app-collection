#!/bin/bash

if [ -z "$INPUTS" ]; then
	echo No inputs specified in INPUTS.
	exit 1;
fi;

if [ -z "$VARIANTS" ]; then
	echo No variants specified in VARIANTS.
	exit 1;
fi;

if [ -z "$RUNS" ]; then
	echo Number of runs to be specified in RUNS.
	exit 1;
fi;

DEBUG=0

while getopts v:i:r:p:d opt; do
    case $opt in
	v)
	    VARIANTS=${OPTARG/,/ /}
	    ;;
	i)
	    INPUTS=$OPTARG
	    ;;
	r)
	    RUNS=$OPTARG
	    ;;
	d)
	    DEBUG=1
	    ;;
	p)
	    PRESERVE=$OPTARG
	    ;;	
	\?)
	    echo Eh?
	    exit 1;
    esac
done;

for i in $INPUTS; do
    echo "** INPUT $i"
    INPUT_LABEL=`basename $i`
    for v in $VARIANTS; do
	for((r = 0; r < $RUNS; r++)); do
	    echo "** RUN $r"
	    echo "** VARIANT $v"
	    if [ $DEBUG -eq 1 ]; then
		echo ./$v $i
	    else
	    	./$v $i
	    fi;
	    [ -z "$PRESERVE" ] || mv $PRESERVE $v-${INPUT_LABEL}-$PRESERVE
	done
    done;
done;

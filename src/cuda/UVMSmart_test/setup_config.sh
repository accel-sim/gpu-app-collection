#!/bin/sh

if [ ! -n "$GPGPUSIM_ROOT" ]; then
	echo "ERROR ** GPGPUSIM_ROOT not set. Run setup_environment first.";
fi

BENCHMARKS=`find . -mindepth 2 -maxdepth 2 -type d -not -path "./data/*"`;

GPGPUSIM_CONFIG=$1
if [ "x$GPGPUSIM_CONFIG" = "x" ]; then 
    echo "";
    echo "Usage: $0 <GPGPU-Sim Config Name | --cleanup>"
    echo "where <GPGPU-Sim Config Name> is one of:"
    ls -1 $GPGPUSIM_ROOT/configs
    echo "";
    exit 0
fi

if [ "x$GPGPUSIM_ROOT" = "x" ]; then 
    GPGPUSIM_ROOT="$PWD/.."
fi

if [ $1 = "--cleanup" ]; then
    echo "Removing existing configs in the following directories:"
    for BMK in $BENCHMARKS; do
        if [ -f $BMK/gpgpusim.config ]; then
            echo "$BMK"
            OLD_ICNT=`awk '/-inter_config_file/ { print $2 }' $BMK/gpgpusim.config`
            rm $BMK/gpgpusim.config $BMK/$OLD_ICNT $BMK/gpuwattch_*.xml
        fi
    done
    exit 0
fi

GPU_CONFIG_FILE=$GPGPUSIM_ROOT/configs/$GPGPUSIM_CONFIG/gpgpusim.config
if [ -f $GPU_CONFIG_FILE ]; then
    echo "Found GPGPU-Sim config file: $GPU_CONFIG_FILE"
else
    echo "Unknown config: $GPGPUSIM_CONFIG"
    exit 0
fi

ICNT_CONFIG=`awk '/-inter_config_file/ { print $2 }' $GPU_CONFIG_FILE`
ICNT_CONFIG=$GPGPUSIM_ROOT/configs/$GPGPUSIM_CONFIG/$ICNT_CONFIG
if [ -f $GPU_CONFIG_FILE ]; then
    echo "Interconnection config file detected: $ICNT_CONFIG"
else
    echo "Interconnection config file not found: $ICNT_CONFIG"
    exit 0
fi
 
GPUWATTCH_CONFIG=$GPGPUSIM_ROOT/configs/$GPGPUSIM_CONFIG/gpuwattch_`echo "$GPGPUSIM_CONFIG" | tr '[:upper:]' '[:lower:]'`.xml
if [ -f $GPUWATTCH_CONFIG ]; then
    echo "Found GPUWATTCH config file: $GPUWATTCH_CONFIG"
else
    echo "Unknown config: $GPUWATTCH_CONFIG"
    exit 0
fi

for BMK in $BENCHMARKS; do
    if [ -f $BMK/gpgpusim.config ]; then
        echo "Existing symbolic-links to config found in $BMK! Skipping... "
    else
        echo "Adding symbolic-links to configuration files for $BMK:"
        ln -v -s $GPU_CONFIG_FILE $BMK
        ln -v -s $ICNT_CONFIG $BMK
        ln -v -s $GPUWATTCH_CONFIG $BMK 
    fi
done

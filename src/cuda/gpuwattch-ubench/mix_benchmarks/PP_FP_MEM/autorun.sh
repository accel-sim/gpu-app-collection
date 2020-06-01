#!/bin/sh

BIN_DIR=../../bin/linux/release 
BIN_NAME=mb_power_phase_fp
DATA_INPUT=
DUMPFILE_NAME=dump_04_02

echo "Excuting $BIN_DIR/$BIN_NAME $DATA_INPUT > $DUMPFILE_NAME"

$BIN_DIR/$BIN_NAME $DATA_INPUT > $DUMPFILE_NAME

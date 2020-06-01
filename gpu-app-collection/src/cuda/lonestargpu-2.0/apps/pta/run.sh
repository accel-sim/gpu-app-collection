#!/bin/bash


if [ $# -lt 1 ]; then
    echo Usage: $0 /path/to/files/input-file-prefix
    exit 0;
fi;

IPFX=$1

NODE=${IPFX}_nodes.txt
CONS=${IPFX}_constraints_after_hcd.txt
HCD=${IPFX}_hcd.txt
SOLN=${IPFX}_correct_soln_001.txt


[ -f "$NODE" ] || echo $NODE does not exist
[ -f "$CONS" ] || echo $CONS does not exist
[ -f "$HCD" ] || echo $HCD does not exist
[ -f "$SOLN" ] || echo $SOLN does not exist

if [ -f "$NODE" ] && [ -f "$CONS" ] && [ -f "$HCD" ] && [ -f "$SOLN" ]; then
    echo ./pta $NODE $CONS $HCD $SOLN 1 1
    ./pta $NODE $CONS $HCD $SOLN 1 1
fi;



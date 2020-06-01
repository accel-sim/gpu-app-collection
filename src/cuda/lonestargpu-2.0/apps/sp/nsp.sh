#!/bin/bash

P=`dirname $0`
SPDIR=$P/sp-1.4/

SPSOL=sp_sol.dat
PARTIAL=partial.cnf
WALKSATOUT=ws_out.dat
WALKSATSOL=ws_sol.dat
TOTALSOL=sol.dat

INPUT=$2

if [ ! -d "$SPDIR" ]; then
    echo "WARNING: sp not found in $SPDIR. Will not run WalkSAT, merge and verify."
fi;

if "$@"; then
    if [ -d $SPDIR ]; then
	if $SPDIR/walksat -seed 1 -solcnf -cutoff 100000000 < $PARTIAL > $WALKSATOUT; then
	    if grep -q 'ASSIGNMENT FOUND' $WALKSATOUT; then
		echo "==> WalkSAT found a solution."
		grep --binary-files=text '^v' $WALKSATOUT | cut -c3- > $WALKSATSOL

		echo "==> Verifying WalkSAT solution ..."
		$SPDIR/verify $WALKSATSOL < $PARTIAL

		echo "==> Merging WalkSAT and SP solution ..."
		$SPDIR/merge $SPSOL $WALKSATSOL > $TOTALSOL
		
		$SPDIR/verify $TOTALSOL < $INPUT
	    fi
	else
	    echo "WalkSAT failed to run."
	fi;
    fi;
else
    if [ -d $SPDIR ]; then
	echo running SP
	$SPDIR/sp -%1 -l $INPUT
    fi;
fi;

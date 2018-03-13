#!/bin/bash

REF=$1
shift;

for f in "$@"; do
    diff -u -s $REF $f
done;
#!/bin/bash

INPUTS="../../inputs/USA-road-d.USA.gr ../../inputs/r4-2e23.gr ../../inputs/rmat22.gr"
VARIANTS="bfs bfs-atomic bfs-merrill bfs-wlw bfs-wla bfs-wlc"
RUNS=3

. ../scripts/run-all-template.sh


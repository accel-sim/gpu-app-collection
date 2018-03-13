#!/bin/bash

awk '/^** RUN/ { run = $3 } 
    /^** VARIANT / {variant = $3} 
    /^** INPUT/ {input = $3}
    $1 == "runtime" {print run, variant, $2, $4, input}' "$@"
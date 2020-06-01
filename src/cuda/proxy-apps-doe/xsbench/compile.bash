#!/bin/bash

# if we need options later, add them here
while getopts "" arg; do
  case $arg in
    *)
      echo "this script has no options, IGNORING and proceeding..."
      ;;
  esac
done

make clean && make

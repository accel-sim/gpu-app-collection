#!/bin/bash
# Download standard graph datasets and generate synthetic graph

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/../../../data_dirs/cuda/H100/graph"
mkdir -p "$DATA_DIR"

# Download karate club (standard benchmark from SuiteSparse Matrix Collection)
if [ ! -f "$DATA_DIR/karate.mtx" ]; then
    echo "Downloading karate.mtx..."
    wget -O "$DATA_DIR/karate.tar.gz" \
      https://suitesparse-collection-website.herokuapp.com/MM/Newman/karate.tar.gz
    cd "$DATA_DIR" && tar -xzf karate.tar.gz && mv karate/karate.mtx . && rm -rf karate karate.tar.gz
fi

# Download netscience (standard benchmark from SuiteSparse Matrix Collection)
if [ ! -f "$DATA_DIR/netscience.mtx" ]; then
    echo "Downloading netscience.mtx..."
    wget -O "$DATA_DIR/netscience.tar.gz" \
      https://suitesparse-collection-website.herokuapp.com/MM/Newman/netscience.tar.gz
    cd "$DATA_DIR" && tar -xzf netscience.tar.gz && mv netscience/netscience.mtx . && rm -rf netscience netscience.tar.gz
fi

# Generate synthetic 100k vertex graph
if [ ! -f "$DATA_DIR/synthetic_100k.mtx" ]; then
    echo "Generating synthetic_100k.mtx..."
    python3 "$SCRIPT_DIR/generate_graph.py" -n 100000 -m 5 -o "$DATA_DIR/synthetic_100k.mtx"
fi

echo "Graph data ready in $DATA_DIR"

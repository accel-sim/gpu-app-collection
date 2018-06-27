export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_ROOT=$BASH_ROOT+"/data_dirs/"

wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/all.gpgpu-sim-app-data.tgz
tar xzvf all.gpgpu-sim-app-data.tgz -C $BASH_ROOT
rm all.gpgpu-sim-app-data.tgz

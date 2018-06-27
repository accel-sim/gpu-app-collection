export BASH_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_ROOT=$BASH_ROOT+"/data_dirs/"

cd $BASH_ROOT && tar czvf all.gpgpu-sim-app-data.tgz ./data_dirs/
scp all.gpgpu-sim-app-data.tgz tgrogers@dynamo.ecn.purdue.edu:~/website/gpgpu-sim/benchmark_data/

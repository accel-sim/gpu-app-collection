export DATA_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
# rodinia-2.0-ft data
RODINIA_DIR=$DATA_ROOT/cuda/rodinia/2.0-ft
PANOTTIA_DIR=$DATA_ROOT/pannotia
DRAGON_DIR=$DATA_ROOT/dragon
PROXY_DIR=$DATA_ROOT/proxy-apps-doe
if [ ! -d $RODINIA_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/rodinia-2.0-ft.tgz
    tar xzvf rodinia-2.0-ft.tgz -C $DATA_ROOT
    rm rodinia-2.0-ft.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/pannotia.tgz
    tar xzvf pannotia.tgz -C $DATA_ROOT
    rm pannotia.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/dragon.tgz
    tar xzvf dragon.tgz -C $DATA_ROOT
    rm dragon.tgz
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/proxy-apps-doe.tgz
    tar xzvf proxy-apps-doe.tgz -C $DATA_ROOT
    rm proxy-apps-doe.tgz
fi

export DATA_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_ROOT+="/data_dirs/"
RODINIA_20_FT_DIR=$DATA_ROOT/cuda/rodinia/2.0-ft
RODINIA_31_DIR=$DATA_ROOT/cuda/rodinia/3.1
PANOTIA_DIR=$DATA_ROOT/pannotia
DRAGON_DIR=$DATA_ROOT/dragon
PROXY_DIR=$DATA_ROOT/proxy-apps-doe
SDK_42_DIR=$DATA_ROOT/cuda/sdk/4.2
ISPASS_DIR=$DATA_ROOT/cuda/ispass-2009
LONESTAR_DIR=$DATA_ROOT/cuda/lonestargpu-2.0

mkdir -p $DATA_ROOT

if [ ! -d $RODINIA_20_FT_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/rodinia-2.0-ft.tgz
    tar xzvf rodinia-2.0-ft.tgz -C $DATA_ROOT
    rm rodinia-2.0-ft.tgz
fi

if [ ! -d $PANOTIA_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/pannotia.tgz
    tar xzvf pannotia.tgz -C $DATA_ROOT
    rm pannotia.tgz
fi

if [ ! -d $DRAGON_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/dragon.tgz
    tar xzvf dragon.tgz -C $DATA_ROOT
    rm dragon.tgz
fi

if [ ! -d $PROXY_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/proxy-apps-doe.tgz
    tar xzvf proxy-apps-doe.tgz -C $DATA_ROOT
    rm proxy-apps-doe.tgz
fi

if [ ! -d $SDK_42_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/sdk-4.2.tgz
    tar xzvf sdk-4.2.tgz -C $DATA_ROOT
    rm sdk-4.2.tgz
fi

if [ ! -d $RODINIA_31_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/rodinia-3.1.tgz
    tar xzvf rodinia-3.1.tgz -C $DATA_ROOT
    rm rodinia-3.1.tgz
fi

if [ ! -d $ISPASS_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/ispass-2009-data.tgz
    tar xzvf ispass-2009-data.tgz -C $DATA_ROOT
    rm ispass-2009-data.tgz
fi

if [ ! -d $LONESTAR_DIR ]; then
    wget https://engineering.purdue.edu/tgrogers/gpgpu-sim/benchmark_data/lonestargpu-2.0.tgz
    tar xzvf lonestargpu-2.0.tgz -C $DATA_ROOT
    rm lonestargpu-2.0.tgz
fi

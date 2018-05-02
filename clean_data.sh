export DATA_ROOT="$( cd "$( dirname "$BASH_SOURCE" )" && pwd )"
DATA_ROOT+="/data_dirs"
rm -fr $DATA_ROOT/cuda
rm -fr $DATA_ROOT/pannotia
rm -fr $DATA_ROOT/dragon
rm -fr $DATA_ROOT/proxy-apps-doe

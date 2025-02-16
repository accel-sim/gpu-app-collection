ORIGINAL_FOLDER=$PWD
BASE_MLPERF_DIR=$GPUAPPS_ROOT/bin/$CUDA_VERSION/release/mlperf_inference/
if [ -d $BASE_MLPERF_DIR ]; then
    cd $BASE_MLPERF_DIR
    if [ ! -d ./mlc ]; then
        echo "WARNING: mlperf_inference folder exists but mlc virtual environment is missing, which is necessary for clearing the mlc cache"
        cd $ORIGINAL_FOLDER
        exit 1
    fi
    . ./mlc/bin/activate
    mlc show cache 2>&1 | tee ./mlc_cache.log
    if [ $(wc -l ./mlc_cache.log | awk '{print $1}') -gt 1 ]; then
        mlc rm cache -f &&
        echo "mlc cache cleaned"
    else
        echo "mlc cache is already empty"
    fi
    deactivate
    cd $ORIGINAL_FOLDER
fi

# The virtual environment created for performing inference is stored in 
# $GPUAPPS_ROOT/bin/$CUDA_VERSION/release/mlperf_inference/, which will 
# be removed during make clean. Without clearing the cache, inference 
# will fail the next time it is performed with a new virtual environment. 
# There could be a more efficient solution to this.
# Data: 2025/2/16

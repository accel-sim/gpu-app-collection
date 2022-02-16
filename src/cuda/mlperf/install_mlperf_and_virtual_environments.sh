#Clone mlcommons inference repo

# exit when any command fails
set -e

mkdir mlperf_inference
cd mlperf_inference
git clone https://github.com/mlcommons/inference.git

cd inference/language/bert
wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py

python3 pyvenvex.py virtual_environment
. virtual_environment/bin/activate

#python --version

#################################################
# At this point we are in the virtual environment
#################################################

# First install loadgen
set +e
cd ../../
git submodule update --init third_party/pybind
cd loadgen
pip install absl-py numpy 	
python setup.py bdist_wheel  
pip install --force-reinstall dist/mlperf_loadgen-*.whl

# Go back to bert, install dependencies and make
cd ../language/bert
pip install onnx==1.6.0 transformers==2.4.0 onnxruntime==1.2.0 numpy==1.18.0
pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install nvidia-pyindex
pip install nvidia-tensorflow
make setup

# Run the benchmark
# python run.py --backend pytorch
# Get out of the virtual environment
deactivate


##########################################
# Setup the virtual environment for GNMT #
##########################################
set -e

cd ../../translation/gnmt/tensorflow/

wget https://gist.githubusercontent.com/vsajip/4673395/raw/0504ce930e6dc6b02e4955a07d91ad462e0ba80b/pyvenvex.py

python3 pyvenvex.py virtual_environment
. virtual_environment/bin/activate

set +e

pip install absl-py numpy 	
cd ../../../loadgen
pip install --force-reinstall dist/mlperf_loadgen-*.whl

cd ../translation/gnmt/tensorflow

# Install tensorflow 1.15
# If we don't get the nvidia tensorflow
# Ampere doesn't work

pip install nvidia-pyindex
pip install nvidia-tensorflow

. download_trained_model.sh
. download_dataset.sh
#. verify_dataset.sh - Pausing this for now, checksums are screwy, blame zenodo

# To run the benchmark
# python run_task.py --run=performance --batch_size=32

deactivate

#Clone mlcommons inference repo

echo "--------------------------------------------------------------"
echo "-                                                            -"
echo "-These workloads are to be used as performance reference only-"
echo "-     Go the the official MLCommons website to obtain        -"
echo "-          instructions to make an official run              -"
echo "-                                                            -"
echo "--------------------------------------------------------------"

# exit when any command fails
set -e

if  test -e "$PWD/mlperf_inference"; then
    echo "mlperf inference directory exists, skipping this part"
else
    echo $PWD/mlperf_inference
    mkdir mlperf_inference
    cd mlperf_inference
    git clone https://github.com/mlcommons/inference.git
    # We need to get a fixed version
    cd inference
    git checkout r1.0
    cd ../..
fi

echo $PWD
cd mlperf_inference/inference
BASE_MLPERF_INFERENCE_DIR=$PWD

if [ -f "$BASE_MLPERF_INFERENCE_DIR/virtual_environment_pytorch/bin/activate" ]; then
    echo "Virtual environment exists; skipping this part"
else
    wget -nc https://gist.githubusercontent.com/cesar-avalos3/11691629846ee072080f76d23fa92f45/raw/98b0bab7e1f45f89915af64462d682b69f4e7b6d/pyvenvex.py
    #################################################
    # Single Pytorch environment for all workloads  #
    #################################################
    python3 pyvenvex.py virtual_environment_pytorch
fi

. virtual_environment_pytorch/bin/activate

INVENV=$(python -c 'import sys; print ("1" if hasattr(sys, "real_prefix") else "0")')
if [ $INVENV != "0" ]; then
    echo "Not running a virtual environment"
    return 1 2>/dev/null
    exit 1
fi

cd language/bert
#python --version

#################################################
# At this point we are in the virtual environment
#################################################

# BERT
 
BERT_DEPEND=$BASE_MLPERF_INFERENCE_DIR/bert_inference_depend.done

if [ -f "$BERT_DEPEND" ]; then
    echo "BERT dependencies fulfilled, skipping"
else
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
    pip install onnx==1.8.1 transformers==2.4.0 tokenization onnxruntime==1.2.0 numpy==1.18.0
    pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
    echo "-----------------------------------------------------------"
    echo " pyindex will try to find a version that can be installed -"
    echo "   therefore errors will be commonplace until it finds a  -"
    echo "                   version that works                     -"
    echo "-----------------------------------------------------------"
    pip install tensorflow
    make setup
    echo "BERT dependencies fulfilled" > $BERT_DEPEND
fi 

# RESNET

cd $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection

RESNET_DEPEND=$BASE_MLPERF_INFERENCE_DIR/resnet_inference_depend.done

if [ -f "$RESNET_DEPEND" ]; then
    echo "RESNET dependencies fulfilled, skipping"
else
    wget https://zenodo.org/record/4588417/files/resnet50-19c8e357.pth
    wget https://zenodo.org/record/2535873/files/resnet50_v1.pb
    wget https://zenodo.org/record/4589637/files/resnet50_INT8bit_quantized.pt
    wget https://zenodo.org/record/2592612/files/resnet50_v1.onnx
    pip install opencv-python
    pip install pycocotools
    pip install future
    python setup.py install
    echo "RESNET dependencies fulfilled" > $RESNET_DEPEND
fi

# 3DUNET

cd $BASE_MLPERF_INFERENCE_DIR/vision/medical_imaging/3d-unet/

UNET_DEPEND=$BASE_MLPERF_INFERENCE_DIR/unet_inference_depend.done

if [ -f "$UNET_DEPEND" ]; then
    echo "RESNET dependencies fulfilled, skipping"
else
    if [[ -z "${DOWNLOAD_DATA_DIR}"]]; then
        make setup
        make preprocess_data
        echo "3DUNET dependencies fulfilled" > $UNET_DEPEND
    else
        echo "######################################################################"
        echo "3d-Unet requires the BRATS dataset to be set via the DOWNLOAD_DATA_DIR env. variable"
        echo "######################################################################"        
    fi
fi

# SSD

cd $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection

SSD_DEPEND=$BASE_MLPERF_INFERENCE_DIR/ssd_inference_depend.done

if [ -f "$SSD_DEPEND" ]; then
    echo "SSD dependencies fulfilled, skipping"
else
    mkdir cocos
    cd cocos
    mkdir cocos-full
    mkdir cocos-300-300
    mkdir cocos-1200-1200
    cd cocos-full
    wget -nc http://images.cocodataset.org/zips/val2017.zip
    wget -nc http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip annotations_trainval2017.zip
    unzip val2017.zip
    cd $BASE_MLPERF_INFERENCE_DIR/tools/upscale_coco
    python upscale_coco.py --inputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-full --outputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-300-300 --size 300 300 --format jpg
    python upscale_coco.py --inputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-full --outputs $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection/cocos/cocos-1200-1200 --size 1200 1200 --format jpg
    cd $BASE_MLPERF_INFERENCE_DIR/vision/classification_and_detection
    python setup.py install
    wget -nc https://zenodo.org/record/3345892/files/tf_ssd_resnet34_22.1.zip?download=1
    wget -nc https://zenodo.org/record/3252084/files/mobilenet_v1_ssd_8bit_finetuned.tar.gz
    wget -nc https://zenodo.org/record/3236545/files/resnet34-ssd1200.pytorch
    echo "SSD dependencies fulfilled" > $SSD_DEPEND
fi

deactivate

################################################
# Setup the Tensorflow 1.5 virtual environment #
################################################

cd $BASE_MLPERF_INFERENCE_DIR

set -e

wget -nc https://gist.githubusercontent.com/cesar-avalos3/11691629846ee072080f76d23fa92f45/raw/98b0bab7e1f45f89915af64462d682b69f4e7b6d/pyvenvex.py

if [ -f "$BASE_MLPERF_INFERENCE_DIR/virtual_environment_tensorflow_1_15/bin/activate" ]; then
    echo "Tensorflow 1.15 virtual env exists, skipping this part"
else
    python3 pyvenvex.py $BASE_MLPERF_INFERENCE_DIR/virtual_environment_tensorflow_1_15
    . $BASE_MLPERF_INFERENCE_DIR/virtual_environment_tensorflow_1_15/bin/activate
    pip install absl-py numpy 	
    cd $BASE_MLPERF_INFERENCE_DIR/loadgen
    pip install --force-reinstall dist/mlperf_loadgen-*.whl
fi

. $BASE_MLPERF_INFERENCE_DIR/virtual_environment_tensorflow_1_15/bin/activate

set +e

# GNMT

GNMT_DEPEND=$BASE_MLPERF_INFERENCE_DIR/gnmt_inference_depend.done

if [ -f "$GNMT_DEPEND" ]; then
    echo "GNMT dependencies installed; skipping this part"
else
    cd $BASE_MLPERF_INFERENCE_DIR/translation/gnmt/tensorflow
    pip install nvidia-pyindex
    pip install nvidia-tensorflow
    . download_trained_model.sh
    . download_dataset.sh
    echo "GNMT dependencies fulfilled" > $GNMT_DEPEND
fi

deactivate

cd $BASE_MLPERF_INFERENCE_DIR/../../

# Install the training workloads

# cd ../../../../../

echo 'Done'
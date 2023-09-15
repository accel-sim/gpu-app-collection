
##################################
#   Install the training part    #
##################################

echo "######################################"
echo "#     Installing MLPerf training     #"
echo "######################################"

set -e

if  test -e "$PWD/mlperf_training"; then
    echo "mlperf inference directory exists, skipping this part"
else
    mkdir mlperf_training
    cd mlperf_training
    git clone https://github.com/mlcommons/training.git
    cd ..
fi

cd mlperf_training
BASE_MLPERF_DIR=$PWD/training

cd $BASE_MLPERF_DIR

##############################
# RNN Translator
##############################

RNN_DEPEND=$BASE_MLPERF_DIR/rnn_training_depend.done

if [ -e "$BASE_MLPERF_DIR/virtual_environment_pytorch/bin/activate" ]; then
    echo "Pytorch virtual env. exists, skipping"
else
    cd $BASE_MLPERF_DIR
    wget https://gist.githubusercontent.com/cesar-avalos3/11691629846ee072080f76d23fa92f45/raw/98b0bab7e1f45f89915af64462d682b69f4e7b6d/pyvenvex.py

    python3 pyvenvex.py virtual_environment_pytorch
    . virtual_environment_pytorch/bin/activate
fi

RNN_DEPEND=$BASE_MLPERF_DIR/rnn_training_depend.done

if [ -f "$RNN_DEPEND" ]; then
    echo "RNN Translation dependencies fulfilled, skipping"
else
    set +e
    . $BASE_MLPERF_DIR/virtual_environment_pytorch/bin/activate
    pip install torch===1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
    pip install sacrebleu
    pip install mlperf_compliance
    cd $BASE_MLPERF_DIR/rnn_translator/
    # Replace any set -e
    sed -i 's/set -e/set +e/' download_dataset.sh
    . download_dataset.sh
    echo "RNN Translation dependencies fulfilled" > $RNN_DEPEND
fi

cd ..

##############################
# Single Stage Detector (SSD)
##############################

SSD_DEPEND=$BASE_MLPERF_DIR/ssd_training_depend.done


if [ -f "$SSD_DEPEND" ]; then
    echo "SSD dependencies fulfilled, skipping"
else
    set +e
    . $BASE_MLPERF_DIR/virtual_environment_pytorch/bin/activate
    cd $BASE_MLPERF_DIR/single_stage_detector
    which python
    # -r requirements doesn't always work 
    #pip install -r requirements.txt

    pip install Cython==0.28.4 mlperf-compliance==0.0.10 cycler==0.10.0 kiwisolver==1.0.1 matplotlib==2.2.2 numpy==1.19.1 Pillow==5.2.0 pyparsing==2.2.0 python-dateutil==2.7.3 pytz==2018.5 six==1.11.0 torchvision==0.2.1 pycocotools
    pip install "git+https://github.com/mlperf/logging.git@1.0.0"
    pip install "git+git://github.com/NVIDIA/apex.git@9041a868a1a253172d94b113a963375b9badd030#egg=apex"

    sed -i 's|/coco|coco/|' download_dataset.sh
    # Second one just in case
    sed -i 's|/coco|coco/|' download_dataset.sh

#    . download_dataset.sh
#    . download_resnet34_backbone.sh
    echo "SSD dependencies fulfilled" > $SSD_DEPEND
fi

##############################
# Object Detection (Mask-RCNN)
##############################

MRCNN_DEPEND=$BASE_MLPERF_DIR/mrcnn_training_depend.done

if [ -f "$MRCNN_DEPEND" ]; then
    echo "SSD dependencies fulfilled, skipping"
else
    set +e
    . $BASE_MLPERF_DIR/virtual_environment_pytorch/bin/activate
    cd $BASE_MLPERF_DIR/object_detection
    which python
    # -r requirements doesn't always work 
    #pip install -r requirements.txt
    pip install opencv-python==4.0.0.21 yacs==0.1.5 ninja==1.8.2.post2
    . download_dataset.sh
#    . download_resnet34_backbone.sh
    echo "Mask-RCNN dependencies fulfilled" > $MRCNN_DEPEND
fi



deactivate

##############################
# BERT
##############################

echo "--------------------------------------"
echo "-   To download the wiki dataset     -"
echo "- DOWNLOAD_BIG_BERT_TRAINING_DATASET -"
echo "- environment variable must be set   -"
echo "--------------------------------------"

# 365 GB of data
cd $BASE_MLPERF_DIR
set -e

wget https://gist.githubusercontent.com/cesar-avalos3/11691629846ee072080f76d23fa92f45/raw/98b0bab7e1f45f89915af64462d682b69f4e7b6d/pyvenvex.py

python3 pyvenvex.py virtual_environment_tensorflow_1_15
. virtual_environment_tensorflow_1_15/bin/activate

set +e

pip install nvidia-pyindex
pip install nvidia-tensorflow



cd $BASE_MLPERF_DIR/language_model/tensorflow/bert/cleanup_scripts
mkdir wiki
cd wiki
    #
    #wget https://dumps.wikimedia.org/enwiki/20200101/enwiki-20200101-pages-articles-multistream.xml.bz2
    #bzip2 -d enwiki-20200101-pages-articles-multistream.xml.bz2
echo "----------------------------------------------------------------------------"
echo "-       Download the enwiki-2020101-pages-articles-multistream and         -"
echo "-        checkpoints from the mlcommons-provided Google Drive link         -" 
echo "- https://drive.google.com/drive/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT -" 
echo "-          as per https://github.com/mlcommons/training/issues/466         -"
echo "----------------------------------------------------------------------------"

cd ..

if [ -f "$PWD/wiki/enwiki-20200101-pages-articles-multistream.xml" ]; then
    git clone https://github.com/attardi/wikiextractor.git
    python wikiextractor/WikiExtractor.py wiki/enwiki-20200101-pages-articles-multistream.xml
    . process_wiki '<text/*/wiki_??'
    python extract_test_set_articles.py
else
    echo "Download the files from the Google Drive link above"
fi

cd $BASE_MLPERF_DIR

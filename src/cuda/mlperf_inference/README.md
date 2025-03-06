# Run with Docker
## Prerequisites
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Pull the Docker Image
Please follow the instructions [here](https://github.com/accel-sim/Dockerfile) to pull the latest Accel-Sim Docker Image.

### Run the container
``` sh
docker run -it --gpus all --mount type=bind,src="PATH_TO_GPU-APP-COLLECTION_HOST",target="PATH_TO_GPU-APP-COLLECTION_CONTAINER" Image_ID
```
**PATH_TO_GPU-APP-COLLECTION_HOST**: The path on the host machine to gpu-app-collection repository.  
**PATH_TO_GPU-APP-COLLECTION_CONTAINER**: The path inside the container where the source will be mounted.  
**Image_ID**: ID of the image pulled from previous step.
### Run the Inference
``` sh
cd PATH_TO_GPU-APP-COLLECTION_CONTAINER/src
source setup_environment
make mlperf_inference
. ../bin/12.8/release/mlperf_inference/inference_mlperf_bert_test.sh
```

# Run with Apptainer
## Build the Apptainer Sandbox
``` sh
apptainer build --sandbox /PATH_TO_SANDBOX ghcr.io/accel-sim/accel-sim-framework:ubuntu-24.04-cuda-12.8
```

## Create an Overlay Image
``` sh
apptainer overlay create --fakeroot --size 1024 /PATH_TO_OVERLAY_IMAGE #e.g. ./overlay/myoverlay.img
```
This creates an overlay image to enable a writable Apptainer container.

## Run the Container
``` sh
 apptainer shell --nv --fakeroot -o /PATH_TO_OVERLAY_IMAG /PATH_TO_SANDBOX
```
Navigate to the gpu-app-collection folder (You might need to bind this into your sandbox container depending on the apptainer configuration)
``` sh
cd PATH_TO_GPU-APP-COLLECTION/src
source setup_environment
make mlperf_inference
. ../bin/12.8/release/mlperf_inference/inference_mlperf_bert_test.sh
```
You might need to manualy delete the /var/lib/dpkg/status-old folder if the dpkg fails due to backup errors.

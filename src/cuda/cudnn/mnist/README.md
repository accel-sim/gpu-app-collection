This folder contains cuDNN MNIST benchmark for  GPGPU-Sim.

If you use this benchmark in your work, please cite:

Jonathan Lew, Deval Shah, Suchita Pati, Shaylin Cattell, Mengchi Zhang, Amruth Sandhupatla, Christopher Ng, Negar Goli, Matthew D Sinclair, Timothy G Rogers, Tor Aamodt, Analyzing Machine Learning Workloads Using a Detailed GPU Simulator, arXiv preprint arXiv:1811.08933, 2018


### Install cuDNN Developer library ###

* Download cuDNN Developer Library. Select the cuDNN version according to the version of cuda runtime version. This benchmark is tested on cuda-8.0 and cuDNN v7.1.4.
* Follow the instructions given on  [https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) to install cuDNN library. (If you don't have sudo permissions, you can extract the debian file and then follow instructions for "Installing from a Tar File")
* You might need to modify Makefile if you are using different version of cuDNN
* [Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)

### Steps to compile and run the benchmark ###

* Modify "CUDA_PATH" in the Makefile
* If you are using different cuDNN version, then you might need to modify "lcudnn_static_v7" in the following line of Makefile:

LIBRARIES += -LFreeImage/lib/$(TARGET_OS)/$(TARGET_ARCH) -LFreeImage/lib/$(TARGET_OS) -lcudart -lcublas_static -lcudnn_static_v7 -lculibos -lfreeimage -lstdc++ -lm -ldl -lpthread

* If you are using different cuDNN version, then you might need to modify "#include <cudnn_v7.h>" in mnistCUDNN.cpp
* Make sure you have sourced the `setup_environment` file inside the GPGPU-Sim dev branch (and built the simulator).
* Make sure you copy appropriate config files for gpgpusim in the current folder

Once these steps are done correctly, following commands should run the benchmark on the simulator

* make clean
* make
* ./mnistCUDNN

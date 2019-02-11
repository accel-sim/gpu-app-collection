If you use this benchmark in your work, please cite:

Jonathan Lew, Deval Shah, Suchita Pati, Shaylin Cattell, Mengchi Zhang, Amruth Sandhupatla, Christopher Ng, Negar Goli, Matthew D Sinclair, Timothy G Rogers, Tor Aamodt, Analyzing Machine Learning Workloads Using a Detailed GPU Simulator, arXiv preprint arXiv:1811.08933, 2018


### Install cuDNN Developer library ###

* This benchmark is tested for cuda-8.0 and cuDNN v7.1.4
* Download cuDNN Developer Library from [https://developer.nvidia.com/rdp/cudnn-archive](https://developer.nvidia.com/rdp/cudnn-archive).
* Follow the instructions given on  [https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html) to install cuDNN library. (If you don't have sudo permissions, you can extract the debian file and then follow the instructions for "Installing from a Tar File")

### Steps to compile and run the benchmark ###

* Modify "CUDA_PATH" in the Makefile
* Set the environment by modifying the PATH and LD_LIBRARY_PATH variables

export PATH=/usr/local/cuda/bin:$PATH

export LD_LIBRARY_PATH=/usr/loca/cuda/lib64:$LD_LIBRARY_PATH

* Make sure you have sourced the `setup_environment` file inside the GPGPU-Sim dev branch (and built the simulator).
* Make sure you have copied appropriate config files for GPGPU-Sim in the current folder

Once these steps are done correctly, following commands should run the benchmark on the simulator

* make clean
* make
* ./mnistCUDNN

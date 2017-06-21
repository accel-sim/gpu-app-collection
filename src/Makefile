
CUDA_VERSION=$(shell nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/')
BINDIR=$(shell pwd)/../bin/$(CUDA_VERSION)
BINSUBDIR=release
SETENV=export BINDIR=$(BINDIR); \
   	export ROOTDIR=$(NVIDIA_COMPUTE_SDK_LOCATION)/C/src/; \
   	export BINSUBDIR=$(BINSUBDIR); \
   	export BOOST_LIB=/usr/lib64; \
	export BOOST_ROOT=/usr/include; \
   	export BOOST_VER=""; \
   	export OPENMPI_BINDIR=/usr/lib64/mpi/gcc/openmpi/bin/; \
   	export ROOTOBJDIR=obj_$(CUDA_VERSION);
noinline?=0

check_environment:
	 @if [ ! -n "$(CUDA_INSTALL_PATH)" ]; then \
		echo "ERROR *** CUDA_INSTALL_PATH not set; please see README.ISPASS-2009"; \
		exit 201; \
	 elif [ ! -d "$(CUDA_INSTALL_PATH)" ]; then \
	 	echo "ERROR *** CUDA_INSTALL_PATH=$(CUDA_INSTALL_PATH) invalid (directory does not exist)."; \
		exit 202; \
	 elif [ ! -n "$(NVIDIA_COMPUTE_SDK_LOCATION)" ]; then \
		echo "ERROR *** NVIDIA_COMPUTE_SDK_LOCATION not set; please see README.ISPASS-2009"; \
		exit 203; \
	 elif [ ! -d "$(NVIDIA_COMPUTE_SDK_LOCATION)" ]; then \
	 	echo "ERROR *** NVIDIA_COMPUTE_SDK_LOCATION=$(NVIDIA_COMPUTE_SDK_LOCATION) invalid (directory does not exist)."; \
		exit 204; \
	 else \
		NVCC_PATH=`which nvcc`; \
		if [ $$? = 1 ]; then \
			echo ""; \
			echo "ERROR ** nvcc (from CUDA Toolkit) was not found in PATH but required to build the ISPASS 2009 benchmarks."; \
			echo "         Try adding $(CUDA_INSTALL_PATH)/bin/ to your PATH environment variable."; \
			echo "         Please also be sure to read README.ISPASS-2009 if you have not done so."; \
			echo ""; \
			exit 206; \
		else\
			CUDA_VERSION=`nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/'`; \
			if [ "$(CUDA_VERSION)"=="2.3" -o "$(CUDA_VERSION)"=="3.1" -o "$(CUDA_VERSION)"=="4.0" -o "$(CUDA_VERSION)"=="4.2" ]; then \
				true;\
			else \
				echo "Only CUDA 2.3, 3.1, 4.0 and 4.0 are supported. Your Cuda Version is $(CUDA_VERSION)"; \
				exit 207;\
			fi\
		fi \
	fi

common: check_environment
	rm -f cuda/common; ln -s $(NVIDIA_COMPUTE_SDK_LOCATION)/C/common cuda/common;

###################################################################################################3
# Rodinia 2.0 Functional Test Stuff
###################################################################################################3
RODINIA_2.0-FT=$(addsuffix -ft,$(shell ls cuda/rodinia/2.0-ft/))
$(RODINIA_2.0-FT): common 
	$(SETENV) make noinline=$(noinline) -C cuda/rodinia/2.0-ft/$(patsubst %-ft,%,$@)

CLEAN_RODINIA_2.0-FT=$(addsuffix _clean,$(RODINIA_2.0-FT))
$(CLEAN_RODINIA_2.0-FT): common
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/$(patsubst %-ft_clean,%,$@)

rodinia_2.0-ft: $(RODINIA_2.0-FT)
clean_rodinia_2.0-ft: $(CLEAN_RODINIA_2.0-FT)

all: rodinia_2.0_ft
cleanall: clean_rodinia_2.0_ft

.PHONY: common


CUDA_VERSION=$(shell nvcc --version | grep release | sed -re 's/.*release ([0-9]+\.[0-9]+).*/\1/')
BINDIR=$(shell pwd)/../bin/$(CUDA_VERSION)
BINSUBDIR=release
SETENV=export BINDIR=$(BINDIR); \
   	export BINSUBDIR=$(BINSUBDIR); \
   	export ROOTOBJDIR=obj_$(CUDA_VERSION);

check_environment:
	 @if [ ! -n "$(CUDA_INSTALL_PATH)" ]; then \
		echo "ERROR *** CUDA_INSTALL_PATH not set; please see README.ISPASS-2009"; \
		exit 201; \
	 elif [ ! -d "$(CUDA_INSTALL_PATH)" ]; then \
	 	echo "ERROR *** CUDA_INSTALL_PATH=$(CUDA_INSTALL_PATH) invalid (directory does not exist)."; \
		exit 202; \
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
				echo "Only CUDA 2.3, 3.1, 4.0 and 4.2 are supported. Your cuda Version is $(CUDA_VERSION)"; \
				exit 207;\
			fi\
		fi \
	fi

###################################################################################################3
# Rodinia 2.0 Functional Test Stuff
###################################################################################################3
rodinia_2.0-ft: check_environment
	$(SETENV) make -C cuda/rodinia/2.0-ft/backprop
	$(SETENV) make -C cuda/rodinia/2.0-ft/bfs
	$(SETENV) make -C cuda/rodinia/2.0-ft/heartwall
	$(SETENV) make -C cuda/rodinia/2.0-ft/hotspot
	$(SETENV) make -C cuda/rodinia/2.0-ft/kmeans
	$(SETENV) make -C cuda/rodinia/2.0-ft/lud
	$(SETENV) make -C cuda/rodinia/2.0-ft/nn
	$(SETENV) make -C cuda/rodinia/2.0-ft/nw
	$(SETENV) make -C cuda/rodinia/2.0-ft/pathfinder
	$(SETENV) make -C cuda/rodinia/2.0-ft/srad

clean_rodinia_2.0-ft:
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/backprop
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/bfs
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/heartwall
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/hotspot
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/kmeans
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/lud
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/nn
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/nw
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/pathfinder
	$(SETENV) make clean -C cuda/rodinia/2.0-ft/srad

all: rodinia_2.0-ft
cleanall: clean_rodinia_2.0-ft

.PHONY: common


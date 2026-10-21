include ../../common/make.config
# Link straight into the shared bin dir under the final benchmark name when
# setup_environment has been sourced, and build in place otherwise. This
# replaces the post-build 'mv' the top-level Makefile used to do, which
# stranded binaries in the source tree whenever a sibling app failed.
RODINIA_OUT ?= $(if $(BINDIR),$(BINDIR)/$(BINSUBDIR),.)


CC := $(CUDA_DIR)/bin/nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_SM80) $(GENCODE_SM86) $(GENCODE_SM90) $(GENCODE_SM100) $(GENCODE_SM103) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75)
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = $(RODINIA_OUT)/pathfinder-rodinia-3.1

release:
	$(CC) -c pathfinder.cu -I$(INCLUDE)
	g++ -fPIC -m64 -o pathfinder pathfinder.o  -L$(CUDA_INSTALL_PATH)/lib64   -lcudart
 
enum:
	$(CC) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debug:
	$(CC) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$$(CUDA_LIB_DIR) 

debugenum:
	$(CC) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f pathfinder

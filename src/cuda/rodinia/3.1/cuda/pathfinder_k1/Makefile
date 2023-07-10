include ../../common/make.config

CC := $(CUDA_DIR)/bin/nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75)
INCLUDE := $(CUDA_DIR)/include

SRC = pathfinder.cu

EXE = pathfinder

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

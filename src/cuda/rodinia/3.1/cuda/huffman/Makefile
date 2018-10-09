NVCC=nvcc
CC = g++

CUDA_INCLUDEPATH=$(CUDA_INSTALL_PATH)/include

NVCC_OPTS=-O3 -lcudart -Xcompiler -m64 -g -G $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62)

GCC_OPTS=-O3 -Wall -Wextra -m64

OBJ = main_test_cu.o stats_logger.o vlc_kernel_sm64huff.o scan.o pack_kernels.o cpuencode.o
SRC = main_test_cu.cu
EXE = pavle

ifdef TESTING 
override TESTING = -DTESTING
endif

ifdef CACHECWLUT
override TESTING = -DCACHECWLUT
endif

pavle: $(OBJ) 
	$(NVCC) $(TESTING) $(CACHECWLUT) $(NVCC_OPTS) $(OBJ) -o $(EXE) 

vlc_kernel_sm64huff.o: vlc_kernel_sm64huff.cu 
	$(NVCC) $(TESTING) -c vlc_kernel_sm64huff.cu $(NVCC_OPTS)

scan.o: scan.cu 
	$(NVCC) -c scan.cu $(NVCC_OPTS)

#cpuencode.o: cpuencode.cu
#	$(NVCC) -c $(NVCC_OPTS) cpuencode.cu

pack_kernels.o: pack_kernels.cu 
	$(NVCC) -c pack_kernels.cu $(NVCC_OPTS)

main_test_cu.o: main_test_cu.cu cutil.h
	$(NVCC) $(TESTING) -c main_test_cu.cu $(NVCC_OPTS) -I $(CUDA_INCLUDEPATH) 

.o:.cpp
	$(CC) ++ $(GCC_OPTS) -c $@ -o $<

clean:
	rm -f *.o $(EXE) 

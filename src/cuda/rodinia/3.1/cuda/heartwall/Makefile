GENCODE_SM10 ?= -gencode=arch=compute_10,code=\"sm_10,compute_10\"
GENCODE_SM13 ?= -gencode=arch=compute_13,code=\"sm_13,compute_13\"
GENCODE_SM20 ?= -gencode=arch=compute_20,code=\"sm_20,compute_20\"
GENCODE_SM30 ?= -gencode=arch=compute_30,code=\"sm_30,compute_30\"
GENCODE_SM35 ?= -gencode=arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_SM50 ?= -gencode=arch=compute_50,code=\"sm_50,compute_50\"
GENCODE_SM60 ?= -gencode=arch=compute_60,code=\"sm_60,compute_60\"
GENCODE_SM62 ?= -gencode=arch=compute_62,code=\"sm_62,compute_62\"

CUDA_VERSION_MAJOR ?= 12
CUDA_VERSION_MINOR ?= 0
ifeq ($(shell test ${CUDA_VERSION_MAJOR} -eq 12 -a ${CUDA_VERSION_MINOR} -ge 8 -o ${CUDA_VERSION_MAJOR} -gt 12; echo $$?), 0)
  GENCODE_SM90 ?= -gencode=arch=compute_90,code=\"sm_90,compute_90\" -gencode=arch=compute_90a,code=\"sm_90a,compute_90a\"
  GENCODE_SM100 ?= -gencode=arch=compute_100,code=\"sm_100,compute_100\" -gencode=arch=compute_100a,code=\"sm_100a,compute_100a\"
else
  GENCODE_SM90 ?=
  GENCODE_SM100 ?=
endif


ifdef OUTPUT
override OUTPUT = -DOUTPUT
endif

# link objects(binaries) together
heartwall: main.o ./AVI/avilib.o ./AVI/avimod.o 
	nvcc  $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM90) $(GENCODE_SM100) main.o ./AVI/avilib.o ./AVI/avimod.o -I/usr/local/cuda/include -lm -o heartwall -lcudart

# compile main function file into object (binary)
main.o: main.cu kernel.cu define.c
	nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM90) $(GENCODE_SM100) $(OUTPUT) $(KERNEL_DIM) main.cu -I./AVI -c -O3 -lcudart

./AVI/avilib.o ./AVI/avimod.o:
	cd AVI; make;

# delete all object files
clean:
	rm -f *.o AVI/*.o heartwall *.linkinfo

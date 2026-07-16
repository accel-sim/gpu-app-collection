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

# Example
# target: dependencies
	# command 1
	# command 2
          # .
          # .
          # .
	# command n

# link objects(binaries) together
myocyte.out:		main.o
	nvcc $(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM90) $(GENCODE_SM100)	 main.o \
				-I/usr/local/cuda/include \
				-L/usr/local/cuda/lib \
				-lm -lcudart \
                                -o myocyte.out

# compile main function file into object (binary)
main.o:	define.c \
				main.cu \
				work.cu \
				solver.cu \
				embedded_fehlberg_7_8.cu \
				master.cu \
				kernel.cu \
				kernel_ecc.cu \
				kernel_cam.cu \
				kernel_fin.cu \
				work_2.cu \
				solver_2.cu \
				embedded_fehlberg_7_8_2.cu \
				kernel_2.cu \
				kernel_ecc_2.cu \
				kernel_cam_2.cu \
				kernel_fin_2.cu \
				file.c \
				timer.c
	nvcc	$(GENCODE_SM10) $(GENCODE_SM13) $(GENCODE_ARCH) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35) $(GENCODE_SM50) $(GENCODE_SM60) $(GENCODE_SM62) $(GENCODE_SM70) $(GENCODE_SM75) $(GENCODE_SM90) $(GENCODE_SM100) main.cu \
				-c -O3 -g

# delete all object files
clean:
	rm -f *.o *.out output.txt

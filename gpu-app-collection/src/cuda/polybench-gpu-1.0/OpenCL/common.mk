OpenCL_SDK=/home/scott/NVIDIA_GPU_Computing_SDK
INCLUDE=-I${OpenCL_SDK}/OpenCL/common/inc
LIBPATH=-L${OpenCL_SDK}/OpenCL/common/lib -L${OpenCL_SDK}/shared/lib
LIB=-lOpenCL -lm
all:
	gcc -O3 ${INCLUDE} ${LIBPATH} ${LIB} ${CFILES} -o ${EXECUTABLE}

clean:
	rm -f *~ *.exe

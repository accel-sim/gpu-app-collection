# Uncomment the appropriate lines to enable build options


# specify CUDA main directory
CUDA_HOME ?= $(CUDA_INSTALL_PATH)

# Precision option
PREC_FLAG = -DDOUBLE
#PREC_FLAG = -DSINGLE

# Using chebyshev coefficients
#CHEBY_FLAG = -DUSE_CHEBY 

# Gzip support
GZIP_FLAG = -DGZIPSUPPORT
LIBS = -lz

FLAGS = -g -O2 $(PREC_FLAG) $(CHEBY_FLAG) $(GZIP_FLAG)

# nvcc flags
# TODO: check if ECX_TARGET is set
# use this only with CUDA 5
#NVCC_FLAGS += -gencode arch=compute_35,code=compute_35
NVCC_FLAGS += -g -gencode arch=compute_20,code=compute_20 --ptxas-options=-v -lineinfo
#NVCC_FLAGS += -gencode arch=compute_20,code=compute_20 -lineinfo
#NVCC_FLAGS += -arch=sm_20 -lineinfo

# example paths set up for cuda, modify as needed
INCS = -I$(CUDA_HOME)/include
LIBS += -L$(CUDA_HOME)/lib64 -lcudart
NVCC = $(CUDA_HOME)/bin/nvcc

INCS += -I. -I./utils -I./prototypes

CU_DEPS = interface.h \
	  compute_force_common.h \
          potentials.h \
	  lj_thread_atom.h \
	  eam_thread_atom.h \
	  eam_cta_box.h \
	  eam_cta_box_agg.h
CU_SRCS = compute_force_kernels.cu \
          timestep_kernels.cu

CPP_DEPS = interface.h \
	   utility.h \
	   utils/domains.h \
	   utils/cheby.h \
	   utils/cmd_parser.h \
	   utils/eam_reader.h \
	   utils/lj_reader.h \
	   utils/sim_reader.h
CPP_SRCS = driver.cpp \
	   utility.cpp \
	   utils/domains.cpp \
	   utils/cheby.cpp \
	   utils/cmd_parser.cpp \
	   utils/lj_reader.cpp \
	   utils/eam_reader.cpp \
	   utils/sim_reader.cpp

OBJS = $(CPP_SRCS:.cpp=.cpp.o) $(CU_SRCS:.cu=.cu.o)

CPP = g++

.SUFFIXES: .cu .cpp .o

all: CoMDCUDA

%.cu.o: %.cu 
	$(NVCC) -c $< $(INCS) $(FLAGS) $(NVCC_FLAGS) $(ECX_FLAGS) -o $@

%.cpp.o: %.cpp $(CPP_DEPS)
	$(CPP) -c $< $(INCS) $(FLAGS) $(ECX_FLAGS) -o $@

CoMDCUDA : $(OBJS)
	$(CPP) -o ./CoMDCUDA $(OBJS) $(LIBS)

clean:
	rm -f *.o utils/*.o ./CoMDCUDA *.ccbak *.os *.s *.ptx

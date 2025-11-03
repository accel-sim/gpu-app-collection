BASE_DIR := $(shell pwd)
BIN_DIR := $(BASE_DIR)/../../../bin/

CUOPTS =  $(CUDA_CPPFLAGS)

CC := nvcc

# CUDA_PATH ?= /use/local/cuda-10.1/

LIB :=

# Generate object file list from SRC (for parallel compilation)
CUDA_SRC_FILES := $(filter %.cu, $(SRC))
CPP_SRC_FILES := $(filter %.cpp, $(SRC))
C_SRC_FILES := $(filter %.c, $(SRC))

# To preserve PTX in multi-step compilation, we have to compile the CUDA source files to .a files
CUDA_LIB_FILES := $(CUDA_SRC_FILES:.cu=.a)

# Host side source files
CPP_OBJECT_FILES := $(CPP_SRC_FILES:.cpp=.o)
C_OBJECT_FILES := $(C_SRC_FILES:.c=.o)
OBJECT_FILES := $(CPP_OBJECT_FILES) $(C_OBJECT_FILES)

# If multiple source files are provided, compile them separately and link
# To preserve PTX in final binary: First create static library, then link to executable
# This avoids nvlink stripping PTX during device linking
release: $(CUDA_LIB_FILES) $(OBJECT_FILES)
	$(CC) $(NVCC_FLAGS) $^ -o $(EXE) -L$(LIB) -lcudart
	mv $(EXE) $(BIN_DIR)

# Pattern rule for compiling individual .cu files to .o files
%.a: %.cu
	$(CC) $(NVCC_FLAGS) $(INCLUDE) $(CUOPTS) --lib $< -o $@

%.o: %.cpp
	$(CC) $(NVCC_FLAGS) $(INCLUDE) $(CUOPTS) -dc $< -o $@

%.o: %.c
	$(CC) $(NVCC_FLAGS) $(INCLUDE) $(CUOPTS) -dc $< -o $@

tuner:
	$(CC) $(NVCC_FLAGS) $(CUOPTS) -DTUNER $(SRC) -o $(EXE) $(INCLUDE) -L$(LIB) -lcudart
	mv $(EXE) $(BIN_DIR)

clean:
	rm -f *.o $(OBJECTS); rm -f $(EXE) $(LIB_FILE)

run:
	./$(EXE)

profile:
	nvprof ./$(EXE)

events:
	nvprof  --events elapsed_cycles_sm ./$(EXE)

profileall:
	nvprof --concurrent-kernels off --print-gpu-trace -u us --metrics all --demangling off --csv --log-file data.csv ./$(EXE)

nvsight:
	nv-nsight-cu-cli --metrics gpc__cycles_elapsed.avg,sm__cycles_elapsed.sum,smsp__inst_executed.sum,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_hit.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,lts__t_sectors_srcunit_tex_op_read.sum,lts__t_sectors_srcunit_tex_op_write.sum,lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_write_lookup_hit.sum,lts__t_sector_op_read_hit_rate.pct,lts__t_sector_op_write_hit_rate.pct,lts__t_sectors_srcunit_tex_op_read.sum.per_second,dram__sectors_read.sum,dram__sectors_write.sum,dram__bytes_read.sum  --csv --page raw ./$(EXE) | tee nsight.csv

ptx:
	cuobjdump -ptx ./$(EXE) | tee ptx.txt

sass:
	cuobjdump -sass ./$(EXE) | tee sass.txt

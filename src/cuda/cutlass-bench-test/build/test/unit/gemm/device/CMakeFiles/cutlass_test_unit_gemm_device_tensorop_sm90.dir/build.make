# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build

# Include any dependencies generated for this target.
include test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/compiler_depend.make

# Include the progress variables for this target.
include test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/progress.make

# Include the compile flags for this target's objects.
include test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/flags.make

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/flags.make
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/includes_CUDA.rsp
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o: /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op.cu
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && /home/tgrogers-raid/a/common/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o -MF CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o.d -x cu -c /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_f16_f16_f16_tensor_op.cu -o CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/flags.make
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/includes_CUDA.rsp
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o: /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && /home/tgrogers-raid/a/common/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o -MF CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o.d -x cu -c /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu -o CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/flags.make
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/includes_CUDA.rsp
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o: /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_s8_s8_s8_tensor_op_s32.cu
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && /home/tgrogers-raid/a/common/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o -MF CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o.d -x cu -c /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_s8_s8_s8_tensor_op_s32.cu -o CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/flags.make
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/includes_CUDA.rsp
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o: /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && /home/tgrogers-raid/a/common/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o -MF CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o.d -x cu -c /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu -o CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/flags.make
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/includes_CUDA.rsp
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o: /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_f32_f32_f32_tensor_op_f32.cu
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CUDA object test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && /home/tgrogers-raid/a/common/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o -MF CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o.d -x cu -c /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device/sm90_gemm_f32_f32_f32_tensor_op_f32.cu -o CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target cutlass_test_unit_gemm_device_tensorop_sm90
cutlass_test_unit_gemm_device_tensorop_sm90_OBJECTS = \
"CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o" \
"CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o" \
"CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o" \
"CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o" \
"CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o"

# External object files for target cutlass_test_unit_gemm_device_tensorop_sm90
cutlass_test_unit_gemm_device_tensorop_sm90_EXTERNAL_OBJECTS = \
"/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/CMakeFiles/cutlass_test_unit_infra.dir/common/filter_architecture.cpp.o" \
"/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/CMakeFiles/cutlass_test_unit_infra_lib.dir/test_unit.cpp.o"

test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f16_f16_f16_tensor_op.cu.o
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_bf16_bf16_bf16_tensor_op_f32.cu.o
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_s8_s8_s8_tensor_op_s32.cu.o
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_tf32_tf32_f32_tensor_op_f32.cu.o
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/sm90_gemm_f32_f32_f32_tensor_op_f32.cu.o
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/CMakeFiles/cutlass_test_unit_infra.dir/common/filter_architecture.cpp.o
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/CMakeFiles/cutlass_test_unit_infra_lib.dir/test_unit.cpp.o
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/build.make
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: lib/libgtest.a
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: /home/tgrogers-raid/a/common/cuda-11.7/lib64/libcudart.so
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: /home/tgrogers-raid/a/common/cuda-11.7/lib64/stubs/libcuda.so
test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90: test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable cutlass_test_unit_gemm_device_tensorop_sm90"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/build: test/unit/gemm/device/cutlass_test_unit_gemm_device_tensorop_sm90
.PHONY : test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/build

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/clean:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && $(CMAKE_COMMAND) -P CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/cmake_clean.cmake
.PHONY : test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/clean

test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/depend:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/unit/gemm/device/CMakeFiles/cutlass_test_unit_gemm_device_tensorop_sm90.dir/depend


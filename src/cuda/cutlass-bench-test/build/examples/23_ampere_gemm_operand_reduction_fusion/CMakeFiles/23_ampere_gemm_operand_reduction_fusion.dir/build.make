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
include examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/compiler_depend.make

# Include the progress variables for this target.
include examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/progress.make

# Include the compile flags for this target's objects.
include examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/flags.make

examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/flags.make
examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/includes_CUDA.rsp
examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o: /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu
examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/examples/23_ampere_gemm_operand_reduction_fusion && /home/tgrogers-raid/a/common/cuda-11.7/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o -MF CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o.d -x cu -c /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/examples/23_ampere_gemm_operand_reduction_fusion/ampere_gemm_operand_reduction_fusion.cu -o CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o

examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target 23_ampere_gemm_operand_reduction_fusion
23_ampere_gemm_operand_reduction_fusion_OBJECTS = \
"CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o"

# External object files for target 23_ampere_gemm_operand_reduction_fusion
23_ampere_gemm_operand_reduction_fusion_EXTERNAL_OBJECTS =

examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/ampere_gemm_operand_reduction_fusion.cu.o
examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/build.make
examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/linkLibs.rsp
examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/objects1
examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion: examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable 23_ampere_gemm_operand_reduction_fusion"
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/examples/23_ampere_gemm_operand_reduction_fusion && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/build: examples/23_ampere_gemm_operand_reduction_fusion/23_ampere_gemm_operand_reduction_fusion
.PHONY : examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/build

examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/clean:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/examples/23_ampere_gemm_operand_reduction_fusion && $(CMAKE_COMMAND) -P CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/cmake_clean.cmake
.PHONY : examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/clean

examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/depend:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/examples/23_ampere_gemm_operand_reduction_fusion /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/examples/23_ampere_gemm_operand_reduction_fusion /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : examples/23_ampere_gemm_operand_reduction_fusion/CMakeFiles/23_ampere_gemm_operand_reduction_fusion.dir/depend

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

# Utility rule file for test_unit_gemm_device_simt_3x.

# Include any custom commands dependencies for this target.
include test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/compiler_depend.make

# Include the progress variables for this target.
include test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/progress.make

test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x: test/unit/gemm/device/cutlass_test_unit_gemm_device_simt_3x
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && ./cutlass_test_unit_gemm_device_simt_3x --gtest_output=xml:test_unit_gemm_device_simt_3x.gtest.xml

test_unit_gemm_device_simt_3x: test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x
test_unit_gemm_device_simt_3x: test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/build.make
.PHONY : test_unit_gemm_device_simt_3x

# Rule to build all files generated by this target.
test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/build: test_unit_gemm_device_simt_3x
.PHONY : test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/build

test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/clean:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device && $(CMAKE_COMMAND) -P CMakeFiles/test_unit_gemm_device_simt_3x.dir/cmake_clean.cmake
.PHONY : test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/clean

test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/depend:
	cd /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/test/unit/gemm/device /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device /home/tgrogers-raid/a/gaur13/accel-sim-updated/accel-sim/accel-sim-framework/gpu-app-collection/src/cuda/cutlass-bench/build/test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : test/unit/gemm/device/CMakeFiles/test_unit_gemm_device_simt_3x.dir/depend


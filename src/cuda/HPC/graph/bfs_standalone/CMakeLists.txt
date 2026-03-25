cmake_minimum_required(VERSION 3.18)
project(bfs_standalone CUDA CXX)

find_package(CUDAToolkit REQUIRED)

# Try to find cuGraph installation
# First check environment variable, then common install locations
if(DEFINED ENV{CUGRAPH_ROOT})
    set(cugraph_ROOT $ENV{CUGRAPH_ROOT})
elseif(EXISTS "$ENV{HOME}/cugraph")
    set(cugraph_ROOT "$ENV{HOME}/cugraph")
else()
    set(cugraph_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../external/cugraph")
endif()

add_executable(bfs_standalone bfs_standalone.cu)
target_compile_features(bfs_standalone PRIVATE cxx_std_17)

# Add include directories for cuGraph and dependencies
target_include_directories(bfs_standalone PRIVATE
    ${cugraph_ROOT}/cpp/include
    ${cugraph_ROOT}/cpp/build/_deps/rmm-src/include
    ${cugraph_ROOT}/cpp/build/_deps/raft-src/cpp/include
    ${cugraph_ROOT}/cpp/build/_deps/cuvs-src/cpp/include
    ${cugraph_ROOT}/cpp/build/_deps/rapids_logger-src/include
)

# Link libraries
target_link_directories(bfs_standalone PRIVATE
    ${cugraph_ROOT}/cpp/build
    ${cugraph_ROOT}/cpp/build/_deps/rmm-build
    ${cugraph_ROOT}/cpp/build/_deps/raft-build
    ${cugraph_ROOT}/cpp/build/_deps/cuvs-build
)

target_link_libraries(bfs_standalone PRIVATE
    CUDA::cudart
    CUDA::cublas
    cugraph
    cugraph_c
)

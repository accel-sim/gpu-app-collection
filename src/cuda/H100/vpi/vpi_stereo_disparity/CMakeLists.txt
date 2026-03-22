cmake_minimum_required(VERSION 3.18)
project(vpi_stereo_disparity CXX)

find_package(vpi QUIET)

if(vpi_FOUND)
    add_executable(vpi_stereo_disparity main.cpp)
    target_link_libraries(vpi_stereo_disparity PRIVATE vpi)
else()
    message(WARNING "VPI not found - skipping vpi_stereo_disparity")
endif()

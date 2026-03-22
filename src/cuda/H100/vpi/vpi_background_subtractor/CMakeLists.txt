cmake_minimum_required(VERSION 3.18)
project(vpi_background_subtractor CXX)

# Find VPI library (optional)
find_package(vpi QUIET)

if(vpi_FOUND)
    add_executable(vpi_background_subtractor main.cpp)
    target_link_libraries(vpi_background_subtractor PRIVATE vpi)
else()
    message(WARNING "VPI not found - skipping vpi_background_subtractor")
endif()

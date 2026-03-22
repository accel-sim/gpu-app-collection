cmake_minimum_required(VERSION 3.18)
project(vpi_orb_feature_detector CXX)

find_package(vpi QUIET)

if(vpi_FOUND)
    add_executable(vpi_orb_feature_detector main.cpp)
    target_link_libraries(vpi_orb_feature_detector PRIVATE vpi)
else()
    message(WARNING "VPI not found - skipping vpi_orb_feature_detector")
endif()

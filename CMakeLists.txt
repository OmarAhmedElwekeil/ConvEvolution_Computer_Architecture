cmake_minimum_required(VERSION 3.31)
project(Project_arch)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx -mfma -march=native")

find_package(OpenMP REQUIRED)

add_executable(Project_arch main.cpp
        loopunrolling.h
        SIMD.h
        OpenMP.h
        OpenMPAndSIMD.h
        naiive.h)

if(OpenMP_CXX_FOUND)
    target_link_libraries(Project_arch PRIVATE OpenMP::OpenMP_CXX)
endif()
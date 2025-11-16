cmake_minimum_required(VERSION 3.22)
project(ginfer CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STARDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -O1")

enable_testing()

include(cmake/3rdparty.cmake)
include(cmake/cuda.cmake)

# global const
set(GINFER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# include directories
include_directories("${GINFER_SRC_DIR}")

option(WITH_CUDA "Build with CUDA" OFF)

add_subdirectory(ginfer)
add_subdirectory(python)

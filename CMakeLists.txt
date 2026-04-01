cmake_minimum_required(VERSION 3.26)
project(ginfer CXX CUDA)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STARDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CUDA_FLAGS_DEBUG "-G -O1")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

enable_testing()

include(cmake/3rdparty.cmake)
include(cmake/cuda.cmake)

# set build type to Release by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# global const
set(GINFER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# include directories
include_directories("${GINFER_SRC_DIR}")

# compile definitions
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_definitions(NDEBUG)
endif()

# options
option(WITH_CUDA "Build with CUDA" OFF)
option(DEBUG_CUDA "Enable CUDA debug flags" OFF)

if(DEBUG_CUDA)
    # 清除所有配置中的优化 flag，避免与 -O1 冲突
    foreach(_cuda_flags_var
        CMAKE_CUDA_FLAGS
        CMAKE_CUDA_FLAGS_DEBUG
        CMAKE_CUDA_FLAGS_RELEASE
        CMAKE_CUDA_FLAGS_RELWITHDEBINFO
        CMAKE_CUDA_FLAGS_MINSIZEREL)
        string(REGEX REPLACE "-O[0-9sg]" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
        # 清除 nvcc 默认注入的 --optimize 长参数（如有）
        string(REGEX REPLACE "--optimize[ =][0-9]+" "" ${_cuda_flags_var} "${${_cuda_flags_var}}")
        string(STRIP "${${_cuda_flags_var}}" ${_cuda_flags_var})
    endforeach()

    # 注入到全局 flag，对所有构建类型生效
    string(APPEND CMAKE_CUDA_FLAGS " -G -O1")

    message(STATUS "CUDA debug mode: -G -O1 enabled, optimizations suppressed")
endif()

add_subdirectory(ginfer)
add_subdirectory(python)

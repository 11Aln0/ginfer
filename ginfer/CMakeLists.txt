function(add_lib_from_path name path)
    file(GLOB SRC_LIST "${path}/*.cpp")
    if(NOT SRC_LIST)
        message(FATAL_ERROR "No source files found in ${path}")
    endif()
    add_library(${name} ${SRC_LIST})
    message(STATUS "Added library ${name}")
endfunction()

add_lib_from_path(memory memory)
target_link_libraries(memory fmt::fmt nlohmann_json::nlohmann_json glog::glog CUDA::cudart)
target_include_directories(memory PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_subdirectory(tests)
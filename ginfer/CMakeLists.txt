function(add_lib_from_path name path)
    file(GLOB SRC_LIST "${path}/*.cpp")
    if(NOT SRC_LIST)
        message(FATAL_ERROR "No source files found in ${path}")
    endif()
    add_library(${name} ${SRC_LIST})
    message(STATUS "Added library ${name}")
endfunction()


set(PATH_LIB_LIST
    "memory|memory"
    "tensor|tensor"
)

foreach(PATH_LIB ${PATH_LIB_LIST})
    string(REPLACE "|" ";" KV ${PATH_LIB})
    list(GET KV 0 LIB_NAME)
    list(GET KV 1 LIB_PATH)
    add_lib_from_path(${LIB_NAME} ${LIB_PATH})
    target_link_libraries(${LIB_NAME} fmt::fmt nlohmann_json::nlohmann_json glog::glog CUDA::cudart)
    target_include_directories(${LIB_NAME} PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endforeach()




add_subdirectory(tests)
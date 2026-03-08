

if(WITH_CUDA)
    file(GLOB CUDA_SRC_LIST "kernels/cuda/*.cu")
    list(APPEND LIB_SRC_LIST ${CUDA_SRC_LIST})
    file(GLOB CUDA_GEMM_SRC_LIST "kernels/cuda/gemm_kernel/*.cu")
    list(APPEND LIB_SRC_LIST ${CUDA_GEMM_SRC_LIST})
    file(GLOB CUDA_PAGE_ATTN_SRC_LIST "kernels/cuda/page_attn_kernel/*.cu")
    list(APPEND LIB_SRC_LIST ${CUDA_PAGE_ATTN_SRC_LIST})
endif()

if(WITH_CPU)
    file(GLOB CPU_SRC_LIST "kernels/cpu/*.cc")
    list(APPEND LIB_SRC_LIST ${CPU_SRC_LIST})
endif()


file(GLOB SRC_LIST "kernels/*.cc")
list(APPEND LIB_SRC_LIST ${SRC_LIST})


add_lib_with_deps(kernel "${LIB_SRC_LIST}" tensor)
add_pathlib_with_deps(op ./ kernel)
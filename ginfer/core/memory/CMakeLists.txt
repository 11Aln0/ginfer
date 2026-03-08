file(GLOB SRCS "*.cc")
file(GLOB CUDA_SRCS "cuda/*.cc")
set(ALL_SRCS ${SRCS} ${CUDA_SRCS})

add_lib_with_deps(memory "${ALL_SRCS}" common)
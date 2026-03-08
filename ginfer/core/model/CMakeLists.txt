file(GLOB TOKENIZER_SRCS "tokenizer/*.cc")
file(GLOB LOADER_SRCS "loader/*.cc")
file(GLOB SRCS "*.cc")
set(ALL_SRCS ${SRCS} ${TOKENIZER_SRCS} ${LOADER_SRCS})

add_lib_with_deps(model "${ALL_SRCS}" layer utils tokenizers_cpp)
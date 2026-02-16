file(GLOB TOKENIZER_SRCS "tokenizer/*.cc")
file(GLOB SRCS "*.cc")
set(ALL_SRCS ${SRCS} ${TOKENIZER_SRCS})

add_lib_with_deps(model "${ALL_SRCS}" layer utils tokenizers_cpp)
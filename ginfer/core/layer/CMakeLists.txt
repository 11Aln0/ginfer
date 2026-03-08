file(GLOB SRC_LIST "*.cc")
file(GLOB TRANSFORMER_SRC_LIST "transformer/*.cc")
set(ALL_SRC_LIST ${SRC_LIST} ${TRANSFORMER_SRC_LIST})

add_lib_with_deps(layer "${ALL_SRC_LIST}" op)
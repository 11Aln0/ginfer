file(GLOB SRC_LIST "*.cc")

add_lib_with_deps(model "${SRC_LIST}" layer)
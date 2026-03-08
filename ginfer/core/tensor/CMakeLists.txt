file(GLOB SRC_LIST "*.cc")
add_lib_with_deps(tensor "${SRC_LIST}" memory)

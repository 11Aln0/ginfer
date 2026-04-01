file(GLOB SRCS "*.cc")

add_lib_with_deps(engine "${SRCS}" common model)
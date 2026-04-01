add_subdirectory(memory)
add_subdirectory(tensor)
add_subdirectory(op)
add_subdirectory(layer)

# 聚合接口库，外部只需链接 core
add_library(core INTERFACE)
target_link_libraries(core INTERFACE
    memory
    tensor
    op
    layer
)

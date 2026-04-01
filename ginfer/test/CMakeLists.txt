function(add_gtest test_name)
    set(lib ${ARGN})
    add_executable(${test_name} ${test_name}.cc test_main.cc)
    target_link_libraries(${test_name} ${lib} 
                          GTest::gtest 
                          glog::glog 
                          -Wl,--whole-archive kernel -Wl,--no-whole-archive
                          )
    add_test(NAME ${test_name} COMMAND ${test_name})
    message(STATUS "Added test ${test_name}")
endfunction()

add_gtest(test_3rd_party fmt::fmt nlohmann_json::nlohmann_json tokenizers_cpp jinja utils)
add_gtest(test_memory memory)
add_gtest(test_tensor memory tensor)
add_gtest(test_op memory tensor op)
add_gtest(test_engine model engine)

# add pybind test
file(GLOB PYBIND_SOURCES "pybind/*.cc")
file(GLOB PYBIND_OP_TEST_SOURCES "op/test_*.cc")
file(GLOB PYBIND_MODEL_TEST_SOURCES "model/test_*.cc")
pybind11_add_module(ginfer_test ${PYBIND_SOURCES} ${PYBIND_OP_TEST_SOURCES} ${PYBIND_MODEL_TEST_SOURCES})
target_link_libraries(ginfer_test PRIVATE 
    pybind11::module 
    core
    model
    -Wl,--whole-archive kernel -Wl,--no-whole-archive
)
set_target_properties(ginfer_test PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)
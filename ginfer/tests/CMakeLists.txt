function(add_gtest test_name lib)
    add_executable(${test_name} ${test_name}.cpp test_main.cpp)
    target_link_libraries(${test_name} ${lib} GTest::gtest)
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

add_gtest(test_3rd_party alloc)

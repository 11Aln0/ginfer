find_package(Python3 REQUIRED COMPONENTS Interpreter)

function(add_python_test test_name)
    add_test(NAME ${test_name}
        COMMAND ${Python3_EXECUTABLE} -m pytest ${CMAKE_CURRENT_SOURCE_DIR}/test/${test_name}.py
    )
    set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python"
    )
    message(STATUS "Added Python test ${test_name}")
endfunction()

add_python_test(test_op)
add_python_test(test_model)
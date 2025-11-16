# CPM
include(cmake/CPM.cmake)
set(CPM_USE_LOCAL_PACKAGES ON)
set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM")

CPMAddPackage("gh:nlohmann/json@3.12.0")
CPMAddPackage("gh:fmtlib/fmt#11.2.0")
CPMAddPackage("gh:google/googletest@1.15.0")
CPMAddPackage("gh:pybind/pybind11@2.13.6")

CPMAddPackage(
    URI "gh:google/glog@0.7.1"
    OPTIONS
        "BUILD_TESTING OFF"  
        "WITH_GFLAGS OFF"
        "WITH_UNWIND OFF"
)

CPMAddPackage("gh:cameron314/concurrentqueue@1.0.4")
if(NOT TARGET concurrentqueue)
  add_library(concurrentqueue INTERFACE)
  target_include_directories(concurrentqueue INTERFACE ${concurrentqueue_SOURCE_DIR})
endif()


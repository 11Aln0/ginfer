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


CPMAddPackage(
  NAME tokenizers_cpp
  GITHUB_REPOSITORY mlc-ai/tokenizers-cpp
  GIT_TAG 34885cfd7b9ef27b859c28a41e71413dd31926f5
  OPTIONS
    "TOKENIZERS_CPP_EXAMPLE OFF"
)

CPMAddPackage(
  NAME jinja
  GITHUB_REPOSITORY 11Aln0/jinja.cpp
  GIT_TAG e69a9b76562a85869cda28ee245cee487298d17a
  OPTIONS
    "JINJA_USE_EXTERNAL_JSON ON"
    "JINJA_BUILD_TESTS OFF"
)
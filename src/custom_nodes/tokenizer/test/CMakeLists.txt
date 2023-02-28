cmake_minimum_required(VERSION 3.10)
project(test)

set(CMAKE_CXX_STANDARD 17)

# Download and install Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

set(UTIL_DIRS
    ../src/             # for model.hpp
    ../../../)          # for custom_node_interface.h

add_executable(tokenization_test tokenization_test.cpp)
target_include_directories(tokenization_test PRIVATE ${UTIL_DIRS})
target_link_libraries(tokenization_test gtest_main tokenizer)

add_executable(detokenization_test detokenization_test.cpp)
target_include_directories(detokenization_test PRIVATE ${UTIL_DIRS})
target_link_libraries(detokenization_test gtest_main detokenizer)

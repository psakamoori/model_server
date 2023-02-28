cmake_minimum_required(VERSION 3.10)
project(tokenizer)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(BLINGFIRE_INSTALL_DIR ${CMAKE_BINARY_DIR}/external/blingfire)

ExternalProject_Add(blingfire
    PREFIX blingfire
    GIT_REPOSITORY https://github.com/microsoft/BlingFire.git
    GIT_TAG master
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BLINGFIRE_INSTALL_DIR}
)

# Download and install Google Test
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

add_library(tokenizer SHARED src/model.cpp src/tokenizer.cpp)
target_include_directories(tokenizer PRIVATE ${BLINGFIRE_INSTALL_DIR}/include)
target_link_libraries(tokenizer
    ${CMAKE_BINARY_DIR}/blingfire/src/blingfire-build/libbingfirtinydll_static.a
    ${CMAKE_BINARY_DIR}/blingfire/src/blingfire-build/libfsaClient.a)
add_dependencies(tokenizer blingfire)

add_library(detokenizer SHARED src/model.cpp src/detokenizer.cpp)
target_include_directories(detokenizer PRIVATE ${BLINGFIRE_INSTALL_DIR}/include)
target_link_libraries(detokenizer
    ${CMAKE_BINARY_DIR}/blingfire/src/blingfire-build/libbingfirtinydll_static.a
    ${CMAKE_BINARY_DIR}/blingfire/src/blingfire-build/libfsaClient.a)
add_dependencies(detokenizer blingfire)

add_executable(tokenization_test test/tokenization_test.cpp)
target_link_libraries(tokenization_test gtest_main tokenizer)

add_executable(detokenization_test test/detokenization_test.cpp)
target_link_libraries(detokenization_test gtest_main detokenizer)

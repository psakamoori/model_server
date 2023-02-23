cmake_minimum_required(VERSION 3.10)
project(tokenizer)

include(ExternalProject)

set(BLINGFIRE_INSTALL_DIR ${CMAKE_BINARY_DIR}/external/blingfire)

ExternalProject_Add(blingfire
    PREFIX blingfire
    GIT_REPOSITORY https://github.com/microsoft/BlingFire.git
    GIT_TAG master
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BLINGFIRE_INSTALL_DIR}
)

add_library(tokenizer SHARED tokenizer.cpp)
target_include_directories(tokenizer PRIVATE ${BLINGFIRE_INSTALL_DIR}/include)
target_link_libraries(tokenizer
    ${CMAKE_BINARY_DIR}/blingfire/src/blingfire-build/libbingfirtinydll_static.a
    ${CMAKE_BINARY_DIR}/blingfire/src/blingfire-build/libfsaClient.a)
add_dependencies(tokenizer blingfire)

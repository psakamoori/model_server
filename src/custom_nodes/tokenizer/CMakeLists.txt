cmake_minimum_required(VERSION 3.10)
project(node)

include(ExternalProject)

# uncomment for logs
# add_compile_definitions(DEBUG)

add_subdirectory(src)
add_subdirectory(test)

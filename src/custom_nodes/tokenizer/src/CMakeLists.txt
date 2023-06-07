#
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
cmake_minimum_required(VERSION 3.10)
project(tokenizer)

include(ExternalProject)

set(CMAKE_CXX_STANDARD 17)

set(BLINGFIRE_INSTALL_DIR ${CMAKE_BINARY_DIR}/external/blingfire)

ExternalProject_Add(blingfire
    PREFIX blingfire
    GIT_REPOSITORY https://github.com/microsoft/BlingFire.git
    GIT_TAG ${BLINGFIRE_SHA}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${BLINGFIRE_INSTALL_DIR}
)

set(BLINGFIRE_STATIC_LIB_DIR ${CMAKE_BINARY_DIR}/src/blingfire/src/blingfire-build)
set(BLINGFIRE_STATIC_LIBS
    ${BLINGFIRE_STATIC_LIB_DIR}/libbingfirtinydll_static.a
    ${BLINGFIRE_STATIC_LIB_DIR}/libfsaClient.a)

ExternalProject_Add(pcre
    PREFIX pcre
    URL http://deb.debian.org/debian/pool/main/p/pcre3/pcre3_8.39.orig.tar.bz2
    URL_HASH SHA1=5e38289fd1b4ef3e8426f31a01e34b6924d80b90
    CMAKE_ARGS -DPCRE_SUPPORT_UNICODE_PROPERTIES=ON -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    INSTALL_COMMAND cmake -E echo "Skipping install step."
)

set(PCRE_STATIC_LIB_DIR ${CMAKE_BINARY_DIR}/src/pcre/src/pcre-build)
set(PCRE_STATIC_LIBS
    ${PCRE_STATIC_LIB_DIR}/libpcre.a)

set(UTIL_DIRS
    ../../common/       # for common/utils.hpp
    ../../../)          # for custom_node_interface.h

add_library(tokenizer SHARED blingfire_model.cpp tokenizer.cpp)
target_include_directories(tokenizer PRIVATE ${BLINGFIRE_INSTALL_DIR}/include ${UTIL_DIRS})
target_link_libraries(tokenizer ${BLINGFIRE_STATIC_LIBS} stdc++fs)
target_compile_options(tokenizer PRIVATE -fstack-protector -fno-omit-frame-pointer -fno-strict-overflow -Wall -Wno-unknown-pragmas -Werror -Wno-error=sign-compare -fno-delete-null-pointer-checks -fwrapv -fstack-clash-protection -Wformat -Wformat-security -Werror=format-security)
add_dependencies(tokenizer blingfire)

add_library(detokenizer SHARED blingfire_model.cpp detokenizer.cpp)
target_include_directories(detokenizer PRIVATE ${BLINGFIRE_INSTALL_DIR}/include ${UTIL_DIRS})
target_link_libraries(detokenizer ${BLINGFIRE_STATIC_LIBS} stdc++fs)
target_compile_options(detokenizer PRIVATE -fstack-protector -fno-omit-frame-pointer -fno-strict-overflow -Wall -Wno-unknown-pragmas -Werror -Wno-error=sign-compare -fno-delete-null-pointer-checks -fwrapv -fstack-clash-protection -Wformat -Wformat-security -Werror=format-security)
add_dependencies(detokenizer blingfire)

add_library(clip_tokenizer SHARED clip_model.cpp clip_tokenizer.cpp)
target_include_directories(clip_tokenizer PRIVATE ${PCRE_STATIC_LIB_DIR} ${UTIL_DIRS})
target_link_libraries(clip_tokenizer ${PCRE_STATIC_LIBS} stdc++fs)
target_compile_options(clip_tokenizer PRIVATE -fstack-protector -fno-omit-frame-pointer -fno-strict-overflow -Wall -Wno-unknown-pragmas -Werror -Wno-error=sign-compare -fno-delete-null-pointer-checks -fwrapv -fstack-clash-protection -Wformat -Wformat-security -Werror=format-security)
add_dependencies(clip_tokenizer pcre)


# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.1)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(OgmaNeo)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

if(DISABLE_THREADING)
    message("Disabling threading")
    add_compile_definitions(KERNEL_NOTHREAD)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include_directories("${PROJECT_SOURCE_DIR}/source")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/source")
set(SOURCES
    "${SOURCE_PATH}/ogmaneo/Helpers.cpp"
    "${SOURCE_PATH}/ogmaneo/SparseCoder.cpp"
    "${SOURCE_PATH}/ogmaneo/Predictor.cpp"
    "${SOURCE_PATH}/ogmaneo/Hierarchy.cpp"
    "${SOURCE_PATH}/ogmaneo/ImageEncoder.cpp"
	"${SOURCE_PATH}/ogmaneo/SparseMatrix.cpp"
)

set(HEADERS
	"${SOURCE_PATH}/ogmaneo/Helpers.h"
    "${SOURCE_PATH}/ogmaneo/SparseCoder.h"
    "${SOURCE_PATH}/ogmaneo/Predictor.h"
    "${SOURCE_PATH}/ogmaneo/Hierarchy.h"
    "${SOURCE_PATH}/ogmaneo/ImageEncoder.h"
	"${SOURCE_PATH}/ogmaneo/SparseMatrix.h"
)

add_library(OgmaNeo ${SOURCES} ${HEADERS})

install(TARGETS OgmaNeo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/"
        DESTINATION include
        FILES_MATCHING PATTERN "*.h*")
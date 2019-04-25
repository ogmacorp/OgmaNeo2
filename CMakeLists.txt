# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.8)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(OgmaNeo)

if(NOT CMAKE_BUILD_TYPE)
    message("CMAKE_BUILD_TYPE not set, setting it to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

include_directories("${PROJECT_SOURCE_DIR}/source")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
find_package(OpenCL REQUIRED)
 
include_directories(${OpenCL_INCLUDE_DIR})

set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/source")
set(SOURCES
    "${SOURCE_PATH}/ComputeSystem.cpp"
    "${SOURCE_PATH}/ComputeProgram.cpp"
    "${SOURCE_PATH}/Helpers.cpp"
    "${SOURCE_PATH}/SparseCoder.cpp"
    "${SOURCE_PATH}/Actor.cpp"
    "${SOURCE_PATH}/Hierarchy.cpp"
)

add_library(OgmaNeo ${SOURCES})

target_link_libraries(OgmaNeo ${OpenCL_LIBRARIES})

install(TARGETS OgmaNeo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/"
        DESTINATION include/ogmaneo
        FILES_MATCHING PATTERN "*.h*")
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

include_directories("${PROJECT_SOURCE_DIR}/source")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}")
 
find_package(OpenCL REQUIRED)
 
include_directories(${OPENCL_INCLUDE_DIR})
 
set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/source")
set(SOURCES
    "${SOURCE_PATH}/system/ComputeSystem.cpp"
    "${SOURCE_PATH}/system/ComputeProgram.cpp"
    "${SOURCE_PATH}/neo/Helpers.cpp"
    "${SOURCE_PATH}/neo/SparseCoder.cpp"
    "${SOURCE_PATH}/neo/Predictor.cpp"
    "${SOURCE_PATH}/neo/Actor.cpp"
    "${SOURCE_PATH}/neo/Hierarchy.cpp"
)

add_library(OgmaNeo SHARED ${SOURCES})

target_link_libraries(OgmaNeo ${OPENCL_LIBRARIES})

install(TARGETS OgmaNeo
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)

install(DIRECTORY "${SOURCE_PATH}/system"
        DESTINATION include/ogmaneo
        FILES_MATCHING PATTERN "*.h*")

install(DIRECTORY "${SOURCE_PATH}/neo"
        DESTINATION include/ogmaneo
        FILES_MATCHING PATTERN "*.h*")
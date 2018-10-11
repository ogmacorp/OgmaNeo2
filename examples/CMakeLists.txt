# ----------------------------------------------------------------------------
#  OgmaNeo
#  Copyright(c) 2016-2018 Ogma Intelligent Systems Corp. All rights reserved.
#
#  This copy of OgmaNeo is licensed to you under the terms described
#  in the OGMANEO_LICENSE.md file included in this distribution.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.1)

project(Test)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

# This is only required for the script to work in the version control
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")
 
find_package(OpenCL REQUIRED)
 
include_directories(${OpenCL_INCLUDE_DIR})

find_package(PythonLibs 2.7 REQUIRED)
 
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(SFML 2 REQUIRED network audio graphics window system)
 
include_directories(${SFML_INCLUDE_DIR})

find_package(OgmaNeo REQUIRED)
 
include_directories(${OGMANEO_INCLUDE_DIR})
 
set(SOURCE_PATH "${PROJECT_SOURCE_DIR}/source")
set(SOURCES
    "${SOURCE_PATH}/Test.cpp"
)

add_executable(Test ${SOURCES})

target_link_libraries(Test ${SFML_LIBRARIES})
target_link_libraries(Test ${OpenCL_LIBRARIES})
target_link_libraries(Test ${PYTHON_LIBRARIES})
target_link_libraries(Test ${OGMANEO_LIBRARIES})

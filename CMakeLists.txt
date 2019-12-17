cmake_minimum_required(VERSION 3.10)

project(PredatorPrey LANGUAGES CUDA CXX)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(PROJECT_SOURCE_DIR ${PROJECT_DIR}/src)
set(SRC ${PROJECT_SOURCE_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_DIR}/run)

set(BOOST_ROOT $ENV{BOOST_ROOT})
set(LD_LIBRARY_PATH ${BOOST_ROOT}/stage/lib)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)

find_package(CUDA)
find_package(PythonLibs 3.7 REQUIRED)
find_package(pybind11)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PYTHON_INCLUDE_DIRS}")
include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${BOOST_ROOT}")
link_directories(${LD_LIBRARY_PATH})

file(GLOB_RECURSE SRC_FILES ${SRC}/launcher.cu) # ${SRC}/*.cu)

cuda_add_library(dna SHARED ${SRC}/module.cpp ${SRC}/launcher.cu)
target_link_libraries(dna  ${PYTHON_LIBRARIES}  cudart)
set_target_properties(dna PROPERTIES PREFIX "")
set_target_properties(dna PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_link_libraries(dna boost_timer)
target_link_libraries(dna boost_chrono)
target_link_libraries(dna boost_filesystem)
target_link_libraries(dna boost_system)

set(CMAKE_VERBOSE_MAKEFILE OFF)
# set(CMAKE_CXX_FLAGS "-v")
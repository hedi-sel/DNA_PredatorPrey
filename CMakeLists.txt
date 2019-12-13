cmake_minimum_required(VERSION 3.10)

project(PredatorPrey LANGUAGES CUDA CXX)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(PROJECT_SOURCE_DIR ${PROJECT_DIR}/src)
set(SRC ${PROJECT_SOURCE_DIR})
set(PYBIND11_PATH ${PROJECT_DIR}/lib/pybind11)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_DIR}/run)
execute_process(COMMAND cd ${CMAKE_BINARY_DIR})

set(BOOST_ROOT $ENV{BOOST_ROOT})
set(LD_LIBRARY_PATH ${BOOST_ROOT}/stage/lib)

# set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CUDA_STANDARD 11)
# set (CMAKE_CUDA_FLAGS "${CMAKE_CXX_FLAGS} -dc")
set(CMAKE_CUDA_COMPILER "nvcc")

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PYBIND11_PATH}/include")
include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${BOOST_ROOT}")
link_directories(${LD_LIBRARY_PATH})

find_package(CUDA)
enable_language(CUDA)
find_package(PythonLibs 2.7 REQUIRED)
add_subdirectory(${PYBIND11_PATH})

file(GLOB_RECURSE sources ${SRC}/*.cu)

# cuda_add_library(mdules STATIC ${sources})
cuda_add_library(mdules STATIC src/launcher.cu)

# target_link_libraries(main ${PYTHON_LIBRARIES} cudart)
set_target_properties(mdules PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
# set_source_files_properties(launcher PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

add_executable(main ${SRC}/main.cpp)

target_link_libraries(main boost_timer)
target_link_libraries(main boost_chrono)
target_link_libraries(main boost_filesystem)
target_link_libraries(main boost_system)
target_link_libraries(main mdules)
set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_VERBOSE_MAKEFILE OFF)
# set(CMAKE_CXX_FLAGS "-v")
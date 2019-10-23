cmake_minimum_required(VERSION 3.0)
project(PredatorPrey LANGUAGES CXX CUDA)

file(MAKE_DIRECTORY output)
file(MAKE_DIRECTORY bin)
execute_process(COMMAND cd ./bin)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR})
set(CMAKE_SOURCE_DIR ${PROJECT_DIR}/src)
set(SRC ${CMAKE_SOURCE_DIR})
set(CMAKE_BINARY_DIR ${PROJECT_DIR}/cmake)
set(BUILD_RPATH ${PROJECT_DIR})
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_DIR}/run)

set(BOOST_ROOT $ENV{BOOST_ROOT})
set(LD_LIBRARY_PATH ${BOOST_ROOT}/stage/lib)

set(CMAKE_CXX_COMPILER "g++")
set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_COMPILER "nvcc")

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${CMAKE_SOURCE_DIR}")
include_directories(${BOOST_ROOT})
link_directories(${LD_LIBRARY_PATH})

file(GLOB sources ${SRC}/main.cpp ${SRC}/PdeSystem/predator_prey_system.cpp ${SRC}/PdeSystem/predator_prey_system_gpu.cpp ${SRC}/PdeSystem/cudaComputer.cu)
add_executable(predatorPrey ${sources})
target_link_libraries(predatorPrey boost_timer)
target_link_libraries(predatorPrey boost_chrono)
target_link_libraries(predatorPrey boost_filesystem)
target_link_libraries(predatorPrey boost_system)
set_property(TARGET predatorPrey PROPERTY CUDA_SEPARABLE_COMPILATION ON)


set(CMAKE_VERBOSE_MAKEFILE OFF)
# set(CMAKE_CXX_FLAGS "-v")

# run shell::cmake -S src/ -B CMak

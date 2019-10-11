cmake_minimum_required(VERSION 3.0)
project(PredatorPrey LANGUAGES CXX)

set(PROJECT_DIR ${CMAKE_SOURCE_DIR}/../)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_DIR}/run)

set(BOOST_ROOT $ENV{BOOST_ROOT})
set(LD_LIBRARY_PATH ${BOOST_ROOT}/stage/lib)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "g++")

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories(${BOOST_ROOT})
link_directories(${LD_LIBRARY_PATH})

add_executable(predatorPrey predatorPrey.cu PdeSystem/predator_prey_system.cpp PdeSystem/predator_prey_system_gpu.cu basic_computation_functions.cu)
target_link_libraries(predatorPrey boost_timer)
target_link_libraries(predatorPrey boost_chrono)
target_link_libraries(predatorPrey boost_filesystem)

# set(CMAKE_VERBOSE_MAKEFILE ON)
# set(CMAKE_CXX_FLAGS "-v")

# run shell::cmake -S src/ -B CMak

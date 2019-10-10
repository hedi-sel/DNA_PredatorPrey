cmake_minimum_required(VERSION 3.0)

# set(CMAKE_VERBOSE_MAKEFILE ON)

project(PredatorPrey LANGUAGES CXX)

# Set the output folder where your program will be created
#set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/../CMake/bin)
set(PROJECT_DIR ${CMAKE_SOURCE_DIR}/../)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_DIR}/run)

set(BOOST_ROOT $ENV{BOOST_ROOT})
set(LD_LIBRARY_PATH ${BOOST_ROOT}/stage/lib)

set (CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_VERBOSE_MAKEFILE ON)


# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories(${BOOST_ROOT})
link_directories(${LD_LIBRARY_PATH})

add_executable(predatorPrey predatorPrey.cpp PdeSystem/predator_prey_system.cpp PdeSystem/predator_prey_system_gpu.cpp basic_computation_functions.cpp)
target_link_libraries(predatorPrey boost_timer)
target_link_libraries(predatorPrey boost_chrono)
target_link_libraries(predatorPrey boost_filesystem)

find_package(OpenMP)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# run shell::cmake -S src/ -B CMak
cmake_minimum_required(VERSION 3.15)

# set(CMAKE_VERBOSE_MAKEFILE ON)

project(PredatorPrey)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(PROJECT_DIR ${CMAKE_SOURCE_DIR}/../)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_DIR}/run)

set(BOOST_ROOT $ENV{BOOST_ROOT})
set(LD_LIBRARY_PATH ${BOOST_ROOT}/stage/lib)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories(${BOOST_ROOT})
link_directories(${LD_LIBRARY_PATH})

add_executable(predatorPrey predatorPrey.cpp)
target_link_libraries(predatorPrey boost_timer)
target_link_libraries(predatorPrey boost_chrono)
target_link_libraries(predatorPrey boost_filesystem)

find_package(OpenMP)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

# run in shell: cmake -S src/ -B CMake/
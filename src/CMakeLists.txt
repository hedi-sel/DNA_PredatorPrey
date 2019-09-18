cmake_minimum_required(VERSION 3.15)

# set(CMAKE_VERBOSE_MAKEFILE ON)

project(PredatorPrey)

# Set the output folder where your program will be created
set(CMAKE_BINARY_DIR ${CMAKE_SOURCE_DIR}/bin)
set(PROJECT_DIR "C:/Users/Hedi/Desktop/Programming/DNA_PredatorPrey")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_DIR}/run)

# The following folder will be included
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("C:/ProgrammingTools/boost_1_70_0/include/boost-1_70/")
link_directories("C:/ProgrammingTools/boost_1_70_0/lib/")

add_executable(predatorPrey predatorPrey.cpp)
target_link_libraries(predatorPrey boost_timer-mgw81-mt-x32-1_70.a)
target_link_libraries(predatorPrey boost_chrono-mgw81-mt-x32-1_70.a)
target_link_libraries(predatorPrey boost_filesystem-mgw81-mt-d-x32-1_70.a)

find_package(OpenMP)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
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
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories("${CMAKE_SOURCE_DIR}")
include_directories("${BOOST_ROOT}")
link_directories(${LD_LIBRARY_PATH})

find_package(CUDA)
find_package(PythonLibs 2.7 REQUIRED)
# add_subdirectory(${PYBIND11_PATH})

file(GLOB_RECURSE sources ${SRC}/*.cu)

# cuda_add_library(mdules STATIC ${sources})
cuda_add_library(PDE_Solver STATIC src/launcher.cu)

# target_link_libraries(main ${PYTHON_LIBRARIES} cudart)
set_target_properties(PDE_Solver PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_link_libraries(PDE_Solver boost_timer)
target_link_libraries(PDE_Solver boost_chrono)
target_link_libraries(PDE_Solver boost_filesystem)
target_link_libraries(PDE_Solver boost_system)
# set_source_files_properties(launcher PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)

# add_executable(main ${SRC}/main.cpp)
# target_link_libraries(main PDE_Solver)
# set_property(TARGET main PROPERTY CUDA_SEPARABLE_COMPILATION ON)

cuda_add_library(dna SHARED src/module.cpp ${SRC}/launcher.cu)
target_link_libraries(dna  ${PYTHON_LIBRARIES}  cudart PDE_Solver)
set_target_properties(dna PROPERTIES PREFIX "")

set_target_properties(dna PROPERTIES CUDA_SEPERABLE_COMPILATION ON)
target_link_libraries(dna boost_timer)
target_link_libraries(dna boost_chrono)
target_link_libraries(dna boost_filesystem)
target_link_libraries(dna boost_system)

set(CMAKE_VERBOSE_MAKEFILE OFF)
# set(CMAKE_CXX_FLAGS "-v")
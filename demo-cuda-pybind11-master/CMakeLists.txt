cmake_minimum_required(VERSION 3.10)
find_package(CUDA)
find_package(PythonLibs 2.7 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_COMPILER "/usr/bin/gcc-6")

cuda_add_library(gpu_library SHARED
  gpu_library.cpp
  gpu_library.cu)

target_link_libraries(gpu_library
  ${PYTHON_LIBRARIES}
  cudart)

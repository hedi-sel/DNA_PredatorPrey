#pragma once

#include <constants.hpp>

#if is2D
__device__ inline dim3 position()
{
    return dim3(threadIdx.z, threadIdx.x + blockDim.x * blockIdx.x, threadIdx.y + blockDim.y * BlockIdx.y);
}
#else
__device__ inline dim3 position()
{
    return dim3(threadIdx.z, threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y, 0);
}
#endif
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <dataStructure/state.h>
#include <utilitary/cudaErrorCheck.h>
#include <utilitary/cudaHelper.h>
#include <constants.hpp>

__device__ inline T square(const T val)
{
    return val * val;
}

__global__ void gaussianFunction(State<T> &x, int species, dim2 sampleSize, T max, dim2 center, int width)
{
    dim3 pos = position();
    pos.x = species;
    if (x.WithinBoundaries(pos.y))
    {
        x(pos) = max * exp(-square((T(pos.y) - center.x) / width)); //- ((is2D) ? pow(pos.z - center.y, 2) : 0))
    }
}

void gaussianMaker(State<T> &x, int species, dim2 sampleSize, T max, dim2 center, int width)
{
    dim3 threads = x.GetThreadDim();
    threads.z = 1;
    gaussianFunction<<<x.GetBlockDim(), x.GetThreadDim()>>>(*x._device, species, sampleSize, max, center, width);
    gpuErrchk(cudaDeviceSynchronize());
}
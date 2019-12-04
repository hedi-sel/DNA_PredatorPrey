#include <iostream>
#include <stdio.h>
#include <math.h>
#include <dataStructure/state.h>
#include <utilitary/cudaErrorCheck.h>
#include <utilitary/cudaHelper.h>

__global__ void gaussianFunction(State<T> &x, int species, int sampleSizeX, T max, int center, int width)
{
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    if (j < x.sampleSizeX)
    {
        x(species, j) = max * exp(-pow(j - center, 2) / T(width * width));
    }
}
__global__ void gaussianFunction2D(State<T> &x, int species, int sampleSizeX, T max, int center, int width)
{
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    if (j < x.sampleSizeX)
    {
        x(species, j) = max * exp(-pow(j - center, 2) / T(width * width));
    }
}

void gaussianMaker(State<T> &x, int species, int sampleSize, int k, T max, int center, int width)
{
    if (is2D)
        gaussianFunction2D<<<x.GetBlockDim(), x.GetThreadDim()>>>(*x._device, species, sampleSize, max, center, width);
    else
        gaussianFunction<<<x.GetBlockDim(), x.GetThreadDim()>>>(*x._device, species, sampleSize, max, center, width);
    gpuErrchk(cudaDeviceSynchronize());

    // m(species, j) = function(j);
    /* 
        if (species == 0 && j >3658 && j< 3661)
            printf("%i %f \n",j, - pow(j - center, 2) / T(width * width)); */
}
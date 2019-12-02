#include <iostream>
#include <stdio.h>
#include <math.h>
#include <dataStructure/state.h>
#include <utilitary/cudaErrorCheck.h>

__global__ void gaussianFunction(State<double> x, int species, int sampleSizeX, double max, int center, int width)
{
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    if (j < x.sampleSizeX)
    {
        x(species, j) = max * exp(-pow(j - center, 2) / double(width * width));
        // printf("theorque %i : %f \n", j, x.data[j]);
    }
}

void gaussianMaker(State<double> &x, int species, int sampleSize, double max, int center, int width)
{

    dim3 threadsPerBlock(32, 32);
    int numBlocks = (x.sampleSizeX * x.sampleSizeY) / (threadsPerBlock.x * threadsPerBlock.y) + 1;

    gaussianFunction<<<numBlocks, threadsPerBlock>>>(x, species, x.sampleSizeX, max, center, width);
    gpuErrchk(cudaDeviceSynchronize());

    // m(species, j) = function(j);
    /* 
        if (species == 0 && j >3658 && j< 3661)
            printf("%i %f \n",j, - pow(j - center, 2) / double(width * width)); */
}
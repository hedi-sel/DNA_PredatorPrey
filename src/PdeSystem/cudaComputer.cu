#include <iostream>
#include "cudaComputer.hpp"
#include <array>
#include <assert.h>


#include "functions.cu"
__global__ void operation(const double* prey, const double* pred, double* diff, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y;
    assert (j < size);
    if((j+1)%size > 1)
        diff[j + size * i] = (i == 0)? devPreyFunction(prey[j], pred[j], devLaplacien(&prey[j]))
                : devPredatorFunction(prey[j], pred[j], devLaplacien( &pred[j]));
    else
        diff[j + size * i] = 0;
}

void compute(const double* prey, const double* pred, double* diff, int size){
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(2, size / (threadsPerBlock.x * threadsPerBlock.y));
    operation<<<numBlocks,threadsPerBlock>>>(prey,pred, diff, size);
    
    /* diff[j] = preyFunction(prey[j%size], pred[j%size], laplacien(&prey[j]));
    diff[j + size] = predatorFunction(prey[j], pred[j], laplacien(&pred[j]));
    for (size_t i = 0; i < 2; ++i)
        diff[size * i] = diff[size * (i+1) - 1] = 0.0; */

}
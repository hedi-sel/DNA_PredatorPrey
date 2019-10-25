#include <iostream>
#include <stdio.h>
#include "cudaComputer.hpp"
#include <array>
#include <assert.h>
#include <functions.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void operation(const double* prey, const double* pred, double* diff, int size)
{
    int i = blockIdx.x;
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y;
    if(j < size && i<2)
        if((j+1)%size > 1)
        diff[j + size * i] = (i == 0)? devPreyFunction(prey[j], pred[j], devLaplacien(&prey[j]))
                : devPredatorFunction(prey[j], pred[j], devLaplacien( &pred[j]));
    else
        diff[j + size * i] = 0;
}

void compute(const double* prey, const double* pred, double* diff, int size){
    dim3 threadsPerBlock(32, 32, 1);
    dim3 numBlocks(2, 1, 1);
    double* devPrey = NULL;
    double* devPred = NULL;
    double* devDiff = NULL;
	cudaMalloc(&devPrey, size*sizeof(double));
	cudaMalloc(&devPred, size*sizeof(double));
	cudaMalloc(&devDiff, 2 * size*sizeof(double));
	cudaMemcpy(devPrey, prey, size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devPred, pred, size*sizeof(double), cudaMemcpyHostToDevice);
    operation<<<numBlocks,threadsPerBlock>>>(devPrey,devPred, devDiff, size);
    cudaMemcpy(diff, devDiff, 2 * size*sizeof(double), cudaMemcpyDeviceToHost);

    
    /* diff[j] = preyFunction(prey[j%size], pred[j%size], laplacien(&prey[j]));
    diff[j + size] = predatorFunction(prey[j], pred[j], laplacien(&pred[j]));
    for (size_t i = 0; i < 2; ++i)
        diff[size * i] = diff[size * (i+1) - 1] = 0.0; */

}
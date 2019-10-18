#include "cudaComputer.hpp"
#include <predPreyFunctions.cu>
#include <array>
#include <assert.h>

__global__ void operation(const double* prey, const double* pred, double* diff, int size)
{
    int j = threadIdx.x + blockIdx.x * blockDim.x;
    assert (j < size*2);
    if((j+1)%size > 1){
        diff[j] = (j<size)? devPreyFunction(prey[j%size], pred[j%size], devLaplacien(&prey[j]))
            : devPredatorFunction(prey[j%size], pred[j%size], devLaplacien( &pred[j]));
    }
    else
        diff[j] = 0;
}

void compute(const double* prey, const double* pred, double* diff, int size){
    double* devPrey = NULL;
    double* devPred = NULL;
    double* devDiff = NULL;
	cudaMalloc(&devPrey, size*sizeof(double));
	cudaMalloc(&devPred, size*sizeof(double));
	cudaMalloc(&devDiff, size*sizeof(double));
	cudaMemcpy(devPrey, prey, size*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(devPred, pred, size*sizeof(double), cudaMemcpyHostToDevice);
    operation<<<128,128>>>(devPrey,devPred, devDiff, size);
    cudaMemcpy(diff, devDiff, size*sizeof(double), cudaMemcpyDeviceToHost);
       
    /* diff[j] = preyFunction(prey[j%size], pred[j%size], laplacien(&prey[j]));
    diff[j + size] = predatorFunction(prey[j], pred[j], laplacien(&pred[j]));
    for (size_t i = 0; i < 2; ++i)
        diff[size * i] = diff[size * (i+1) - 1] = 0.0; */
    
}
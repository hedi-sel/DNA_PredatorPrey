#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <string>
#include <stdio.h>

#include <assert.h>

#include "predator_prey_systems_cuda.hpp"
#include <constants.hpp>
#include <functions.h>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__device__ dim3 position()
{
    return dim3(blockIdx.x, threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y);
}
__global__ void differentiate(double* x, double* dxdt, int im, int jm, double t, double dt)
{
    int i = blockIdx.x;
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y;
    //i = position().x; j = position().y;
    if((j+1)%jm > 1)
        dxdt[j + jm * i] = (i == 0)? devPreyFunction(x[j], x[j + jm], devLaplacien(&x[j]))
                : devPredatorFunction(x[j], x[j+jm], devLaplacien( &x[j+jm]));
    else
        dxdt[j + jm * i] = 0;
    // dxdt[j + jm * i] = 1;
}
__global__ void addStep(double* x, double* dxdt,int im, int jm, double t, double dt)
{
    int i = blockIdx.x;
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y;
    // printf("%d", dxdt[j + jm * i]);
    x[j + jm * i] += dxdt[j + jm * i] * dt;
}

prey_predator_iterator::prey_predator_iterator(double *x, int im, int jm, bool print){
    this->im = im;
    this->jm = jm;
    this->doPrint = print;
    gpuErrchk(  cudaMalloc(&this->x, im * jm * sizeof(double)) );
    gpuErrchk(  cudaMalloc(&this->dxdt, im * jm * sizeof(double)) );
    gpuErrchk(  cudaMemcpy(this->x, x, im * jm * sizeof(double), cudaMemcpyHostToDevice) );
    //this->snapPeriod = snapPeriod;
    dim3 threadsPerBlock(10, 30);
    dim3 numBlocks(2);//, im * jm / (threadsPerBlock.x * threadsPerBlock.y));
   // cudaMemcpy(xDevice, x, im*jm*sizeof(double), cudaMemcpyHostToDevice);
   if(doPrint) 
       printer(0.0);

}
prey_predator_iterator::~prey_predator_iterator(){
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaFree(dxdt));
};

void prey_predator_iterator::iterate(double t, double dt){
    // dim3 threadsPerBlock(32, 32);
    stepCount += 1;
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(im * jm / (threadsPerBlock.x * threadsPerBlock.y));//, im * jm / (threadsPerBlock.x * threadsPerBlock.y));
    differentiate<<<numBlocks,threadsPerBlock>>>(x, dxdt, im, jm, t, dt);
    addStep<<<numBlocks,threadsPerBlock>>>(x, dxdt, im, jm, t, dt);
    if(doPrint && stepCount%500 == 0)
        printer(t);
    //TODO check why x doesn't change
}

void prey_predator_iterator::printer(double t){
    dim3 threadsPerBlock(32,32);
    dim3 numBlocks(im * jm / (threadsPerBlock.x * threadsPerBlock.y));//, im * jm / (threadsPerBlock.x * threadsPerBlock.y));
    double *xHost = new double[im * jm];
    cudaMemcpy(xHost, x, im*jm*sizeof(double), cudaMemcpyDeviceToHost);

    std::ostringstream stream;
    stream << GpuOutputPath << "/state_at_t=" << t << "s.dat";
    std::ofstream fout(stream.str());
    fout << im << "\t" << jm << "\n";
    for (size_t i = 0; i < im*jm; ++i)
    {
        fout << i/jm << "\t" << i%jm << "\t" << xHost[i] << "\n";
    }
}


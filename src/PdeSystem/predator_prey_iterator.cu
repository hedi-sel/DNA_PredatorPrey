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

__device__ int position()
{
    return threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
}

__device__ double differentiate(double* x, int nSpecies, int sampleSize, double t, int pos)
{
    if(pos < nSpecies*sampleSize){
        if((pos+1)%sampleSize > 1)
            return (pos < sampleSize)? devPreyFunction(x[pos], x[pos + sampleSize], devLaplacien(&x[pos]))
                    : devPredatorFunction(x[pos%sampleSize], x[pos], devLaplacien( &x[pos]));
        else
            return 0; 
    }
    else
        return 0;
}

__global__ void rungeKutta4Stepper(double* x, double* dxdt,int nSpecies, int sampleSize, double t, double dt)
{
    int pos = position();    
    if(pos > nSpecies*sampleSize)
        return;
    double k1 = dt * differentiate(x , nSpecies, sampleSize, t, pos);
    x[pos]+=k1/2.0;
    __syncthreads();
    double k2 = dt * differentiate(x, nSpecies, sampleSize, t + dt/2.0, pos);
    x[pos]+=(k2 - k1)/2.0;    
    __syncthreads();
    double k3 = dt * differentiate(x, nSpecies, sampleSize, t + dt/2.0, pos);    
    x[pos]+=k3 - k1/2.0;    
    __syncthreads(); 
    double k4 = dt * differentiate(x, nSpecies, sampleSize, t + dt, pos);    
    x[pos] += (k1 + 2*k2 + 2*k3 + k4)/6.0;

}

prey_predator_iterator::prey_predator_iterator(double *x, int nSpecies, int sampleSize, double t0, bool print){
    this->nSpecies = nSpecies;
    this->sampleSize = sampleSize;
    this->doPrint = print;
    this->t = t0;
    gpuErrchk(  cudaMalloc(&this->x, nSpecies * sampleSize * sizeof(double)) );
    gpuErrchk(  cudaMalloc(&this->dxdt, nSpecies * sampleSize * sizeof(double)) );
    gpuErrchk(  cudaMemcpy(this->x, x, nSpecies * sampleSize * sizeof(double), cudaMemcpyHostToDevice) );
   
    if(doPrint) {
        printer(0.0);
        nextPrint += printPeriod;
    }

}
prey_predator_iterator::~prey_predator_iterator(){
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaFree(dxdt));
};

void prey_predator_iterator::iterate(double dt){
    t+=dt;
    dim3 threadsPerBlock(32,32);
    int numBlocks = (nSpecies * sampleSize + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    assert(numBlocks*32*32 > nSpecies*sampleSize);
    rungeKutta4Stepper<<<numBlocks,threadsPerBlock>>>(x, dxdt, nSpecies, sampleSize, t, dt);
    if(doPrint && t>=nextPrint){
        printer(t);
        nextPrint += printPeriod;
    }
    //TODO check why x doesn't change
}

void prey_predator_iterator::printer(double t){
    double *xHost = new double[nSpecies * sampleSize];
    cudaMemcpy(xHost, x, nSpecies*sampleSize*sizeof(double), cudaMemcpyDeviceToHost);

    std::ostringstream stream;
    stream << GpuOutputPath << "/state_at_t=" << t << "s.dat";
    std::ofstream fout(stream.str());
    fout << nSpecies << "\t" << sampleSize << "\n";
    for (size_t i = 0; i < nSpecies*sampleSize; ++i)
    {
        fout << i/sampleSize << "\t" << i%sampleSize << "\t" << xHost[i] << "\n";
    }
}


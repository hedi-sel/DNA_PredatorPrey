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
__device__ double differentiate(double* x, double* dxdt, int im, int jm, double t, int pos)
{
    if(pos < im*jm){
        if((pos+1)%jm > 1){
            dxdt[pos] = (pos < jm)? devPreyFunction(x[pos], x[pos + jm], devLaplacien(&x[pos]))
                    : devPredatorFunction(x[pos%jm], x[pos], devLaplacien( &x[pos]));
            return dxdt[pos];
        }
        else{
            dxdt[pos] = 0;
            return 0;
        }
    }
    else
        return 0;
    // dxdt[j + jm * i] = 1;
}

__device__ double diff(double* x, int im, int jm, double t, int pos)
{
    if(pos < im*jm){
        if((pos+1)%jm > 1)
            return (pos < jm)? devPreyFunction(x[pos], x[pos + jm], devLaplacien(&x[pos]))
                    : devPredatorFunction(x[pos%jm], x[pos], devLaplacien( &x[pos]));
        else
            return 0; 
    }
    else
        return 0;
    // dxdt[j + jm * i] = 1;
}

__global__ void rungeKutta4Stepper(double* x, double* dxdt,int im, int jm, double t, double dt)
{
    int pos = position();    
    if(pos > im*jm)
        return;
    differentiate(x, dxdt, im, jm, t, pos);
    double k1 = dt * diff(x , im, jm, t, pos);
    x[pos]+=k1/2.0;
    __syncthreads();
    double k2 = dt * diff(x, im, jm, t + dt/2.0, pos);
    x[pos]+=(k2 - k1)/2.0;    
    __syncthreads();
    double k3 = dt * diff(x, im, jm, t + dt/2.0, pos);    
    x[pos]+=k3 - k1/2.0;    
    __syncthreads(); 
    double k4 = dt * diff(x, im, jm, t + dt, pos);    
    x[pos] += (k1 + 2*k2 + 2*k3 + k4)/6.0;

}

prey_predator_iterator::prey_predator_iterator(double *x, int im, int jm, bool print){
    this->im = im;
    this->jm = jm;
    this->doPrint = print;
    gpuErrchk(  cudaMalloc(&this->x, im * jm * sizeof(double)) );
    gpuErrchk(  cudaMalloc(&this->dxdt, im * jm * sizeof(double)) );
    gpuErrchk(  cudaMemcpy(this->x, x, im * jm * sizeof(double), cudaMemcpyHostToDevice) );
   
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
    int numBlocks = (im * jm + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    assert(numBlocks*32*32 > im*jm);
    rungeKutta4Stepper<<<numBlocks,threadsPerBlock>>>(x, dxdt, im, jm, t, dt);
    if(doPrint && t>=nextPrint){
        printer(t);
        nextPrint += printPeriod;
    }
    //TODO check why x doesn't change
}

void prey_predator_iterator::printer(double t){
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


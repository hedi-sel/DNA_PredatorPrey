#include <iostream>
#include "predator_prey_systems_cuda.hpp"
#include <assert.h>
#include "functions.cu"

__shared__ double xDevice[size1 * size2];
__shared__ double *dxdtDevice[size1 * size2];
__global__ void differentiate(int im, int jm, double t, double dt)
{
    int i = blockIdx.x;
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y;
    if((j+1)%jm > 1)
        dxdtDevice[j + jm * i] = (i == 0)? devPreyFunction(xDevice[j], xDevice[j + jm], devLaplacien(&xDevice[j]))
                : devPredatorFunction(xDevice[j], xDevice[j], devLaplacien( &xDevice[j+jm]));
    else
        dxdtDevice[j + jm * i] = 0;
}
__global__ void addStep(int im, int jm, double t, double dt)
{
    int i = blockIdx.x;
    int j = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.y * blockDim.x * blockDim.y;
    xDevice[j + jm * i] = dxdtDevice[j + jm * i] * dt;
}

prey_predator_iterator::prey_predator_iterator(double *x, int im, int jm, double snapPeriod = 0.0){
    this->im = im;
    this->jm = jm;
    this->x = x;
    this->snapPeriod = snapPeriod;
    cudaMemcpy(xDevice, x, im*jm*sizeof(double), cudaMemcpyHostToDevice);
}
prey_predator_iterator::~prey_predator_iterator(){
};

void prey_predator_iterator::iterate(double t, double dt){
    // dim3 threadsPerBlock(32, 32);
    dim3 threadsPerBlock(10, 30);
    dim3 numBlocks(2);//, im * jm / (threadsPerBlock.x * threadsPerBlock.y));
    differentiate<<<numBlocks,threadsPerBlock>>>(im, jm, t, dt);
    addStep<<<numBlocks,threadsPerBlock>>>(im, jm, t, dt);
    printer(t,dt,1.0);
    printer(t,dt,0.0);
    printer(t,dt, 9.0);
    
}

void prey_predator_iterator::printer(double t, double dt, double tp){
    if(t >= tp && t<tp+dt){
	    cudaMemcpy(x, xDevice, im*jm*sizeof(double), cudaMemcpyDeviceToHost);
        for (int i = 0; i<im*jm; i++){
            std::cout << x[i] << " ";
        }
        std::cout << std::endl;
        std::cout << std::endl;
    }

}
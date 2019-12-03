#include <stdio.h>
#include <utilitary/functions.h>
#include <assert.h>
#include <utilitary/cudaErrorCheck.h>

__device__ dim3 position()
{
    return dim3(threadIdx.z, threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y, 0);
}

__device__ double differentiate(State<double> &x, double t, dim3 pos)
{
    if ((pos.y + 1) % x.sampleSizeX > 1)
    {
        return (pos.x == 0) ? devPreyFunction(x(pos), x(pos.x + 1, pos.y, pos.z), devLaplacien(&x(pos), x))
                            : devPredatorFunction(x(pos.x - 1, pos.y, pos.z), x(pos), devLaplacien(&x(pos), x));
    }
    else
        return 0;
}

__global__ void rungeKutta4StepperDev(State<double> &x, double t, double dt)
{
    dim3 pos = position();
    if (!x.WithinBoundaries(pos.y))
        return;

    double k1 = dt * differentiate(x, t, pos);
    __syncthreads();
    x(pos) += k1 / 2.0;
    __syncthreads();
    double k2 = dt * differentiate(x, t + dt / 2.0, pos);
    __syncthreads();
    x(pos) += (k2 - k1) / 2.0;
    __syncthreads();
    double k3 = dt * differentiate(x, t + dt / 2.0, pos);
    __syncthreads();
    x(pos) += k3 - k2 / 2.0;
    __syncthreads();
    double k4 = dt * differentiate(x, t + dt, pos);
    __syncthreads();
    x(pos) += -k3 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
}
/* 
__global__ void print(State<double> x)
{
    dim3 pos = position();
    if (pos.y < x.sampleSizeX)
    {
        printf("%i %i %i: %f \n", pos.x, pos.y, pos.z, x(pos));
    }
} */

void rungeKutta4Stepper(State<double> &x, double t, double dt)
{
    dim3 threadsPerBlock(BLOCK_SIZE / 2, BLOCK_SIZE, 2);
    assert(threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z <= BLOCK_SIZE * BLOCK_SIZE);
    int numBlocks = (x.GetSize() + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    assert(numBlocks * BLOCK_SIZE * BLOCK_SIZE > x.GetSize());
    rungeKutta4StepperDev<<<numBlocks, threadsPerBlock>>>(*x._device, t, dt);
    gpuErrchk(cudaDeviceSynchronize());
}

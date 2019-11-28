#include <stdio.h>
#include <utilitary/functions.h>
#include <assert.h>

__device__ int position(State<double> &x)
{
    //TODO make this better
    return threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
    /* printf("milestone4 %i %i %b /n", rawPos, x.GetSize(), rawPos > x.GetSize());
    if (rawPos > x.GetSize())
        return dim3(0, 0, 0);
    return dim3(rawPos % x.nSpecies, rawPos / x.nSpecies, 0); */
}

__device__ double differentiate(State<double> &x, double t, dim3 pos)
{
    if ((pos.x + 1) % x.GetSize() > 1)
    {
        return (pos.x == 0) ? devPreyFunction(x(pos), x(pos.x + 1, pos.y, pos.z), devLaplacien(&x(pos)))
                            : devPredatorFunction(x(pos.x - 1, pos.y, pos.z), x(pos), devLaplacien(&x(pos)));
    }
    else
        return 0;
}

__global__ void rungeKutta4StepperDev(State<double> &x, double t, double dt)
{
    int rawPos = position(x);
    dim3 pos(rawPos % x.nSpecies, rawPos / x.nSpecies, 0); /* 
    dim3 pos = position(x);
    printf("%i", pos.x);
    if (pos.x == NULL)
        return; */
    printf("milestone \n");
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

void rungeKutta4Stepper(State<double> &x, double t, double dt)
{
    dim3 threadsPerBlock(32, 32);
    int numBlocks = (x.GetSize() + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    assert(numBlocks * BLOCK_SIZE * BLOCK_SIZE > x.GetSize());
    rungeKutta4StepperDev<<<numBlocks, threadsPerBlock>>>(x, t, dt);
    cudaDeviceSynchronize();
}

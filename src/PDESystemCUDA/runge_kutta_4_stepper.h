#include <dataStructure/state.h>
#include <stdio.h>
#include <utilitary/functions.h>
#include <assert.h>
#include <utilitary/cudaErrorCheck.h>
#include <utilitary/cudaHelper.h>

__device__ T differentiate(State<T> &x, T t, dim3 pos)
{
    if ((pos.y + 1) % x.sampleSizeX > 1)
    {
        return (pos.x == 0) ? devPreyFunction(x(pos), x(pos.x + 1, pos.y, pos.z), devLaplacien(&x(pos), x))
                            : devPredatorFunction(x(pos.x - 1, pos.y, pos.z), x(pos), devLaplacien(&x(pos), x));
    }
    else
        return 0;
}

__global__ void rungeKutta4StepperDev(State<T> &x, T t, T dt)
{
    dim3 pos = position();
    if (!x.WithinBoundaries(pos.y))
        return;

    T k1 = dt * differentiate(x, t, pos);
    __syncthreads();
    x(pos) += k1 / 2.0;
    __syncthreads();
    T k2 = dt * differentiate(x, t + dt / 2.0, pos);
    __syncthreads();
    x(pos) += (k2 - k1) / 2.0;
    __syncthreads();
    T k3 = dt * differentiate(x, t + dt / 2.0, pos);
    __syncthreads();
    x(pos) += k3 - k2 / 2.0;
    __syncthreads();
    T k4 = dt * differentiate(x, t + dt, pos);
    __syncthreads();
    x(pos) += -k3 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
}
/* 
__global__ void print(State<T> x)
{
    dim3 pos = position();
    if (pos.y < x.sampleSizeX)
    {
        printf("%i %i %i: %f \n", pos.x, pos.y, pos.z, x(pos));
    }
} */

void rungeKutta4Stepper(State<T> &x, T t, T dt)
{
    rungeKutta4StepperDev<<<x.GetBlockDim(), x.GetThreadDim()>>>(*x._device, t, dt);
    gpuErrchk(cudaDeviceSynchronize());
}

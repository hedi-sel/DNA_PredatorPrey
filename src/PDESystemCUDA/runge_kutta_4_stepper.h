#include <dataStructure/state.h>
#include <stdio.h>
#include <utilitary/functions.h>
#include <assert.h>
#include <utilitary/cudaErrorCheck.h>
#include <utilitary/cudaHelper.h>

__device__ T differentiate(State<T> &x, T t, dim3 pos, const T& xpos)
{
    if ((pos.y + 1) % x.sampleSizeX > 1)
    {
        return (pos.x == 0) ? devPreyFunction(xpos, x(pos.x + 1, pos.y, pos.z), devLaplacien(&xpos, x))
                            : devPredatorFunction(x(pos.x - 1, pos.y, pos.z), xpos, devLaplacien(&xpos, x));
    }
    else
        return 0;
}

__global__ void rungeKutta4StepperDev(State<T> &x, T t, T dt)
{
    dim3 pos = position();
#if is2d
    if (!x.WithinBoundaries(pos.y, pos.z))
        return;
#else
    if (!x.WithinBoundaries(pos.y))
        return;
#endif
    T &xpos = x(pos);
    T k1 = dt * differentiate(x, t, pos, xpos);
    __syncthreads();
    xpos += k1 / 2.0;
    __syncthreads();
    T k2 = dt * differentiate(x, t + dt / 2.0, pos, xpos);
    __syncthreads();
    xpos += (k2 - k1) / 2.0;
    __syncthreads();
    T k3 = dt * differentiate(x, t + dt / 2.0, pos, xpos);
    __syncthreads();
    xpos += k3 - k2 / 2.0;
    __syncthreads();
    T k4 = dt * differentiate(x, t + dt, pos, xpos);
    __syncthreads();
    xpos += -k3 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
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

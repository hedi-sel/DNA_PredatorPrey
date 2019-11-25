#include <stdio.h>
#include <utilitary/functions.h>
#include <assert.h>
__device__ int position()
{
    return threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
}

__device__ double differentiate(double *x, int nSpecies, int sampleSize, double t, int pos)
{
    if ((pos + 1) % sampleSize > 1)
    {
        if (!(&x[pos])[0] == x[pos])
            printf("not the right position at %i at time %d \n", pos, t);

        if (!(&x[pos])[1] == x[pos + 1])
            printf("not the right position at %i at time %d \n", pos, t);

        if (!(&x[pos])[-1] == x[pos - 1])
            printf("not the right position at %i at time %d \n", pos, t);
        return (pos < sampleSize) ? devPreyFunction(x[pos], x[pos + sampleSize], devLaplacien(&x[pos]))
                                  : devPredatorFunction(x[pos % sampleSize], x[pos], devLaplacien(&x[pos]));
    }
    else
        return 0;
}

__global__ void rungeKutta4StepperDev(double *x, double *dxdt, int nSpecies, int sampleSize, double t, double dt)
{
    int pos = position();
    if (pos > nSpecies * sampleSize)
        return;

    //if (x[pos] > 1.0 || x[pos] < 0.0)
    //printf("t= %d  pos = %i  x= %d  lapl= %d \n", t, pos, x[pos], devLaplacien(&x[pos]));
    double k1 = dt * differentiate(x, nSpecies, sampleSize, t, pos);
    __syncthreads();
    x[pos] += k1 / 2.0;
    __syncthreads();
    double k2 = dt * differentiate(x, nSpecies, sampleSize, t + dt / 2.0, pos);
    __syncthreads();
    x[pos] += (k2 - k1) / 2.0;
    __syncthreads();
    double k3 = dt * differentiate(x, nSpecies, sampleSize, t + dt / 2.0, pos);
    __syncthreads();
    x[pos] += k3 - k2 / 2.0;
    __syncthreads();
    double k4 = dt * differentiate(x, nSpecies, sampleSize, t + dt, pos);
    __syncthreads();
    x[pos] += -k3 + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
    
}

void rungeKutta4Stepper(double *x, double *dxdt, int nSpecies, int sampleSize, double t, double dt)
{
    dim3 threadsPerBlock(32, 32);
    int numBlocks = (nSpecies * sampleSize + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    assert(numBlocks * 32 * 32 > nSpecies * sampleSize);
    rungeKutta4StepperDev<<<numBlocks, threadsPerBlock>>>(x, dxdt, nSpecies, sampleSize, t, dt);
}

#include <stdio.h>

#include <functions.h>

__device__ int position()
{
    return threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * blockDim.x * blockDim.y;
}

__device__ double differentiate(double *x, int nSpecies, int sampleSize, double t, int pos)
{
    if (pos < nSpecies * sampleSize)
    {
        if ((pos + 1) % sampleSize > 1)
            return (pos < sampleSize) ? devPreyFunction(x[pos], x[pos + sampleSize], devLaplacien(&x[pos]))
                                      : devPredatorFunction(x[pos % sampleSize], x[pos], devLaplacien(&x[pos]));
        else
            return 0;
    }
    else
        return 0;
}

__global__ void rungeKutta4Stepper(double *x, double *dxdt, int nSpecies, int sampleSize, double t, double dt)
{
    int pos = position();
    if (pos > nSpecies * sampleSize)
        return;
    double k1 = dt * differentiate(x, nSpecies, sampleSize, t, pos);
    x[pos] += k1 / 2.0;
    __syncthreads();
    double k2 = dt * differentiate(x, nSpecies, sampleSize, t + dt / 2.0, pos);
    x[pos] += (k2 - k1) / 2.0;
    __syncthreads();
    double k3 = dt * differentiate(x, nSpecies, sampleSize, t + dt / 2.0, pos);
    x[pos] += k3 - k1 / 2.0;
    __syncthreads();
    double k4 = dt * differentiate(x, nSpecies, sampleSize, t + dt, pos);
    x[pos] += (k1 + 2 * k2 + 2 * k3 + k4) / 6.0;
}


#include <math.h>
#include <dataStructure/state.h>
#include <constants.hpp>
#include "cudaHelper.h"
#include "cudaErrorCheck.h"

#pragma once

//Normalize our dataset by max value
//This code is slow and unoptimized for now
#if USE_DOUBLE
__device__ void atomicMax(double *address, double val)
{
    unsigned long long int *address_as_ull = (unsigned long long int *)address;
    unsigned long long int old = *address_as_ull, assumed;
    do
    {
        assumed = old;
        old = atomicCAS(
            address_as_ull, assumed,
            __double_as_longlong((__longlong_as_double(assumed) > val) ? __longlong_as_double(assumed) : val));
    } while (assumed != old);
    *address = __longlong_as_double(old);
}
#endif
__global__ void FindMax(State<T> &x, State<T> &localMaxes)
{
    dim3 pos = position();
    if (!x.WithinBoundaries(pos.y, pos.z))
        return;

    localMaxes(pos) = x(pos);

    int scannedLength = 1;
    T &localMax = localMaxes(pos);
    while (scannedLength < x.sampleSizeX)
    {
        if (pos.y % (scannedLength * 2) != 0)
            return;
        if (x.WithinBoundaries(pos.y + scannedLength, pos.z))
        {
            T neighb = localMaxes(pos.x, pos.y + scannedLength, pos.z);
            localMax = (localMax > neighb) ? localMax : neighb;
        }
        __syncthreads();
        scannedLength *= 2;
    }
    scannedLength = 1;
    while (scannedLength < x.sampleSizeY)
    {
        if (pos.z % (scannedLength * 2) != 0)
            return;
        if (x.WithinBoundaries(pos.y, pos.z + scannedLength))
        {
            T neighb = localMaxes(pos.x, pos.y, pos.z + scannedLength);
            localMax = (localMax > neighb) ? localMax : neighb;
        }
        __syncthreads();
        scannedLength *= 2;
    }
    /* 
    __shared__ localMax;
    if (threadIdx.x + threadIdx.y > 0)
        return;
    for (int i = 0; i < blockDim.x; i++)
        for (int j = 0; j < blockDim.y; j++)
            if (maxValue[threadIdx.z]>x(
    atomicMax(&maxValue[pos.x], x(pos)); */
}

__global__ void Divide(State<T> &x, State<T> &localMaxes)
{
    dim3 pos = position();
    if (x.WithinBoundaries(pos.y, pos.z))
        x(pos) /= localMaxes(pos.x, 0, 0);
}

void Normalize(State<T> &x)
{
    State<T> localMaxes(nSpecies, sampleSize, true);
    FindMax<<<x.GetBlockDim(), x.GetThreadDim()>>>(*x._device, *localMaxes._device);
    gpuErrchk(cudaDeviceSynchronize());
    
    Divide<<<x.GetBlockDim(), x.GetThreadDim()>>>(*x._device, *localMaxes._device);
    gpuErrchk(cudaDeviceSynchronize());
}
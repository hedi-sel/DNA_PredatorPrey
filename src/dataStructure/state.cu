#include <stdio.h>
#include <assert.h>

#include "utilitary/cudaErrorCheck.h"
#include "state.h"
#include <constants.hpp>

template <typename T>
__host__ State<T>::State(const int nSp, const dim2 samSize, bool isDevice, T *sourceData)
    : State<T>(nSp, samSize.x, samSize.y, isDevice, sourceData){};

template <typename T>
__host__ State<T>::State(const int nSp, const int samSizX, const int samSizY, bool isDevice, T *sourceData)
    : nSpecies(nSp), sampleSizeX(samSizX), sampleSizeY(samSizY),
      isDeviceData(isDevice), data(data)
{
    if (data == nullptr)
    {
        MemAlloc();
    }
    else
    {
        printf("wtf you doin here ");
    }
}

template <typename T>
__host__ void State<T>::MemAlloc()
{
    if (isDeviceData)
    {
        gpuErrchk(cudaMalloc(&data, GetSize() * sizeof(T)));
    }
    else
    {
        data = new T[GetSize()];
    }
    gpuErrchk(cudaMalloc(&_device, sizeof(State<T>)));
    gpuErrchk(cudaMemcpy(_device, this, sizeof(State<T>), cudaMemcpyHostToDevice));
}

template <typename T>
__device__ __host__ State<T>::State(State<T> &state, bool copyToOther)
    : nSpecies(state.nSpecies), sampleSizeX(state.sampleSizeX), sampleSizeY(state.sampleSizeY), isDeviceData((copyToOther) ? !state.isDeviceData : state.isDeviceData),
      data((copyToOther) ? nullptr : state.data)
{
#ifdef __CUDA_ARCH__
#else
    if (copyToOther)
    {
        MemAlloc();
        if (isDeviceData)
        {
            gpuErrchk(cudaMemcpy(data, state.GetRawData(), GetSize() * sizeof(T),
                                 cudaMemcpyHostToDevice));
        }
        else
        {
            gpuErrchk(cudaMemcpy(data, state.GetRawData(), GetSize() * sizeof(T),
                                 cudaMemcpyDeviceToHost));
        }
    }

#endif
}

template <typename T>
__device__ __host__ T &State<T>::operator()(int s, int x, int y)
{
    return data[s * sampleSizeX * sampleSizeY + x * sampleSizeY + y];
}

template <typename T>
__device__ __host__ T &State<T>::GetElementSafe(int s, int x, int y)
{
    if (s < 0 || s > nSpecies)
    {
        printf("Invalid species: %i\n", s);
        return data[0];
    }
    if (OnEdge(x, y))
    {
        if (x == -1)
            return operator()(s, 0, y);
        if (x == sampleSizeX)
            return operator()(s, x - 1, y);
        if (y == sampleSizeY)
            return operator()(s, x, y - 1);
        if (y == -1)
            return operator()(s, x, 0);
        else
            printf("Error: The programmer is a moron\n");
    }
    if (!WithinBoundaries(x, y))
    {
        printf("Invalid position: %i %i\n", x, y);
        return data[0];
    }
    return operator()(s, x, y);
}

template <typename T>
__device__ __host__ T &State<T>::operator()(int p)
{
    return data[p];
}

template <typename T>
__device__ __host__ T &State<T>::operator()(dim3 position)
{
    return this->operator()(position.x, position.y, position.z);
}

template <typename T>
__device__ __host__ const T &State<T>::operator()(const T *el, int x, int y) const
{
    return (el)[x * sampleSizeY + y];
}

template <typename T>
__device__ __host__ bool State<T>::WithinBoundaries(int x, int y)
{
    return !(x < 0 || x >= sampleSizeX || y < 0 || y >= sampleSizeY);
}
template <typename T>
__device__ __host__ bool State<T>::WithinBoundaries(int x)
{
    return !(x < 0 || x >= sampleSizeX);
}

template <typename T>
__device__ __host__ bool State<T>::OnEdge(int x, int y)
{
    return (x == -1 || x == sampleSizeX || y == -1 || y == sampleSizeY);
}

template <typename T>
__device__ __host__ T *State<T>::GetRawData()
{
    return data;
}

template <typename T>
__device__ __host__ int State<T>::GetSize()
{
    return nSpecies * sampleSizeX * sampleSizeY;
}

template <typename T>
__device__ __host__ bool State<T>::Is2D()
{
    return sampleSizeY > 1 && sampleSizeX > 1;
}

template <typename T>
__device__ __host__ dim3 State<T>::GetThreadDim()
{
    return dim3(BLOCK_SIZE / nSpecies, BLOCK_SIZE, nSpecies);
}

template <typename T>
__device__ __host__ dim3 State<T>::GetBlockDim()
{
    dim3 thread = GetThreadDim();
    int blockSize = thread.x * thread.y * thread.z;
    return dim3((GetSize() + blockSize - 1) / blockSize, 1, 1);
}

template <typename T>
__host__ State<T>::~State()
{
    if (isDeviceData)
    {
        gpuErrchk(cudaFree(data));
    }
    else
        delete[] data;
}

template class State<T>;

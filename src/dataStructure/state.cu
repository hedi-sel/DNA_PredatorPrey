#include "state.h"
#include <stdio.h>
#include <assert.h>

template <typename T>
__host__ State<T>::State(const int nSp, const int samSizX, const int samSizY, bool isDevice)
    : nSpecies(nSp), sampleSizeX(samSizX), sampleSizeY(samSizY),
      subSampleSizeX(samSizX), subSampleSizeY(samSizY), isDeviceData(isDevice)
{
    if (isDevice)
    {
        cudaMalloc(&data, GetSize() * sizeof(T));
    }
    else
    {
        data = new T[GetSize()];
    }
}

template <typename T>
__host__ State<T>::State(State<T> &state, bool isDevice)
    : State(state.nSpecies, state.sampleSizeX, state.sampleSizeY, isDevice)
{
    assert(state.isDeviceData == false);
    if (isDevice)
        cudaMemcpy(data, state.GetRawData(), GetSize() * sizeof(T),
                   cudaMemcpyHostToDevice);
    else
    {
        throw "Cannot make these kind of copies yet";
    }
}

template <typename T>
__device__ __host__ T &State<T>::operator()(int s, int x, int y)
{
    if (s < 0 || s > nSpecies)
    {
        printf("Invalid species: %i\n", s);
        return tdefVal;
    }
    if (x == -1 || x == sampleSizeX || y == sampleSizeY || y == -1)
        return tdefVal;
    if (x < -1 || x > sampleSizeX || y > sampleSizeY || y < -1)
    {
        printf("Invalid position: %i %i\n", x, y);
        return tdefVal;
    }
    return data[s * sampleSizeX * sampleSizeY + x * sampleSizeY + y];
}
template <typename T>
__device__ __host__ T &State<T>::operator()(int s, int x)
{
    return this->operator()(s, x, 0);
}

template <typename T>
__device__ __host__ T &State<T>::operator()(dim3 position)
{
    return this->operator()(position.x, position.y, position.z);
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
__host__ State<T>::~State()
{
    delete[] data;
}

template __device__ __host__ class State<double>;
template __device__ __host__ class State<float>;
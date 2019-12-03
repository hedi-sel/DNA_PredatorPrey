#include "state.h"
#include <stdio.h>
#include <assert.h>
#include "utilitary/cudaErrorCheck.h"

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
        if (state.isDeviceData)
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
__device__ __host__ T &State<T>::operator()(int p)
{
    if (p < 0 || p >= GetSize())
    {
        printf("Invalid raw position: %i\n", p);
        return tdefVal;
    }
    return data[p];
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
__device__ __host__ State<T>::~State()
{
    if (isDeviceData)
    {
        //        gpuErrchk(cudaFree(data));
    }
    else
        delete[] data;
}

template class State<double>;
template class State<float>;
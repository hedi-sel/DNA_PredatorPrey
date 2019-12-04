#pragma once

#include "dim2.h"

template <typename T>
class State
{
    // Data structure that captures the state of the system
    // Can handle 1D and 2D
private:
    __host__ void MemAlloc();
    T *data;

public:
    const int nSpecies;
    const int sampleSizeX;
    const int sampleSizeY;

    const bool isDeviceData;

    State<T> *_device;

    // Constructor, set bool = true to create data in device memory
    __host__ State(int, dim2, bool = false, T *data = nullptr);
    __host__ State(int, int, int = 1, bool = false, T *data = nullptr);
    // Copy constructor, set bool = true in case you're copying data host <-> device
    // Usable inside the device for device <-> device copies
    __device__ __host__ State(State<T> &state, bool = false);

    // Element Readers
    __device__ __host__ T &operator()(int, int, int = 0);
    __device__ __host__ T &operator()(dim3);
    // Raw data reader (unsafe)
    __device__ __host__ T &operator()(int);
    // A safer element reader that checks boundaries
    __device__ __host__ T &GetElementSafe(int, int, int = 0);
    // Very unsafe use! Get the element next to the given pointer
    __device__ __host__ const T &operator()(const T *, int, int = 0) const;

    //Checks if the given position is within boundaries of the dataset
    __device__ __host__ bool WithinBoundaries(int, int);
    __device__ __host__ bool WithinBoundaries(int);
    //Checks if the given position is outside but adjascent to the dataset
    __device__ __host__ bool OnEdge(int, int = 0);

    //Get data array
    __device__ __host__ T *GetRawData();

    //Get size of the data array
    __device__ __host__ int GetSize();

    //return true if the state is a 2D one
    __device__ __host__ bool Is2D();

    //Kernel running Helper
    __device__ __host__ dim3 GetThreadDim();
    __device__ __host__ dim3 GetBlockDim();

    //Destructor
    __host__ ~State();
};
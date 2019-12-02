#pragma once
template <typename T>
class State
{
    // Data structure that captures the state of the system
    // Can handle 1D and 2D
private:
    T tdefVal;

    double dx;
    double dt;

    __host__ void MemAlloc();

public:
    const int nSpecies;
    const int sampleSizeX;
    const int sampleSizeY;
    const int subSampleSizeX;
    const int subSampleSizeY;

    T *data;

    bool isDeviceData;

    // Constructor, set bool = true to create data in device memory
    __host__ State(int, int, int = 1, bool = false, T *data = nullptr);
    // Copy constructor, set bool = true in case you're copying data from host to device
    __device__ __host__ State(State<T> &state, bool = false);

    // 2D element reader
    __device__ __host__ T &operator()(int, int, int);
    // 1D element reader
    __device__ __host__ T &operator()(int, int);
    //Universal Rawdata reader
    __device__ __host__ T &operator()(int);

    __device__ __host__ T &operator()(dim3);

    //Get data array
    __device__ __host__ T *GetRawData();

    __device__ __host__ int GetSize();

    //Destructor
    __device__ __host__ ~State();
};
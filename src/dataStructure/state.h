#pragma once
template <typename T>
class State
{
    // Data structure that captures the state of the system
    // Can handle 1D and 2D
private:
    T tdefVal;

    // const double dx;
    // const double dt;

    __host__ void MemAlloc();
    T *data;

public:
    const int nSpecies;
    const int sampleSizeX;
    const int sampleSizeY;

    const bool isDeviceData;

    State<T> *_device;

    // Constructor, set bool = true to create data in device memory
    __host__ State(int, int, int = 1, bool = false, T *data = nullptr);
    // Copy constructor, set bool = true in case you're copying data host <-> device
    // Usable inside the device for device <-> device copies
    __device__ __host__ State(State<T> &state, bool = false);

    // 2D element reader
    __device__ __host__ T &operator()(int, int, int);
    __device__ __host__ T &operator()(dim3);
    // 1D element reader
    __device__ __host__ T &operator()(int, int);
    //Universal Rawdata reader
    __device__ __host__ T &operator()(int);

    //Get data array
    __device__ __host__ T *GetRawData();

    __device__ __host__ int GetSize();

    //Destructor
    __device__ __host__ ~State();
};
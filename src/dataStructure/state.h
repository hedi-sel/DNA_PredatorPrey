#pragma once
template <typename T>
class State
{
    // Data structure that captures the state of the system
    // Can handle 1D and 2D
private:
    T *data;

    T tdefVal;

    double dx;
    double dt;

public:
    const int nSpecies;
    const int sampleSizeX;
    const int sampleSizeY;
    const int subSampleSizeX;
    const int subSampleSizeY;

    bool isDeviceData;

    // Constructor, set bool = true to create data in device memory
    State(int, int, int = 1, bool = false);
    // Copy constructor, set bool = true to make a device copy
    State(State<T>&, bool = false);

    // 2D element reader
    __device__ __host__ T &operator()(int, int, int);
    // 1D element reader
    __device__ __host__ T &operator()(int, int);

    __device__ __host__ T &operator()(dim3);

    //Get data array
    T *GetRawData();

    __device__ __host__ int GetSize();

    //Destructor
    ~State();
};
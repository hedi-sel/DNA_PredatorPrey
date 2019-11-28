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
    
    // 2D constructors
    State(int, int, int = 1);
    // 1D Constructor
    // State(int, int);

    // 2D element reader
    __device__ __host__ T &operator()(int, int, int);
    // 1D element reader
    __device__ __host__ T &operator()(int, int);

    //Get data array
    T *getRawData();

    //Destructor
    ~State();
};
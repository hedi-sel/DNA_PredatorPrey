#include "state.h"
#include <stdio.h>

template <typename T>
State<T>::State(int nSp, int samSizX, int samSizY)
    : nSpecies(nSp), sampleSizeX(samSizX), sampleSizeY(samSizY),
      subSampleSizeX(samSizX), subSampleSizeY(samSizY)
{
    data = new T[nSp * samSizX * samSizY];
}

template <typename T>
T &State<T>::operator()(int s, int x, int y)
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
T &State<T>::operator()(int s, int x)
{
    return this->operator()(s, x, 0);
}

template <typename T>
T *State<T>::getRawData()
{
    return data;
}

template <typename T>
State<T>::~State()
{
    delete[] data;
}

template class State<double>;
template class State<float>;
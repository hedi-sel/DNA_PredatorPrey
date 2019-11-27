#include "state.h"

template <typename T>
State<T>::State(int nSp, int samSizX, int samSizY)
{
    nSpecies = nSp;
    sampleSizeX = samSizX;
    sampleSizeY = samSizY;

    data = new T[nSp * samSizX * samSizY];
}

template <typename T>
State<T>::State(int nSp, int samSizX)
{
    State(nSp, samSizX, 1);
}

template <typename T>
T &State<T>::operator()(int s, int x, int y)
{
    return data[s * sampleSizeX * sampleSizeY + x * sampleSizeY + y];
}

template <typename T>
T &State<T>::operator()(int s, int x)
{
    return this->operator()(s, 0, x);
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
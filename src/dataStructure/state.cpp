#include "state.hpp"

typedef T;

State<T>::State(int nSp, int samSizX, int samSizY)
{
    nSpecies = nSp;
    sampleSizeX = samSizX;
    sampleSizeY = samSizY;

    data = new T[nSp * samSizX * samSizY];
}

State<T>::State(int nSp, int samSizX)
{
    State(nSp, samSizX, 1);
}

T &State<T>::operator()(int s, int x, int y)
{
    return data[s * sampleSizeX * sampleSizeY + x * sampleSizeY + y];
}

T &State<T>::operator()(int s, int x)
{
    return this->operator()(s, 0, x);
}

T *State<T>::getRawData(){
    return data;
}

State<T>::~State(){
    delete[] data;
}
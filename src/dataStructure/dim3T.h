#pragma once

template <typename T>
struct dim
{
    T x;
    T y;
    T z;
    dim<T>(T x, T y = 0, T z = 0) : x(x), y(y), z(z){};
};
#pragma once

#include <dataStructure/state.h>
#include <constants.hpp>
#include <stdio.h>

__device__ inline T devLaplacien(const T *pos, const State<T> &x)
{
#if is2D
    return (-2 * pos[0] + x(pos, 1, 0) + x(pos, -1, 0)) / (dx * dx) +
           (-2 * pos[0] + x(pos, 0, 1) + x(pos, 0, -1)) / (dy * dy);
#else
    return (-2 * pos[0] + x(pos, 1) + x(pos, -1)) / (dx * dx);
#endif
}

__device__ inline T devPreyFunction(const T n, const T p, const T d2n)
{
    return g * n * (1 - B * g * n) - p * n - l * delta * n / (1 + p) + dn * d2n;
}
__device__ inline T devPredatorFunction(const T n, const T p, const T d2p)
{
    return n * p - delta * p / (1 + p) + dp * d2p;
}
/*
    T preyFunctionTaylored(T n, T p, T d2n) const
    {
        return g * n * (1 - n / K) - (1 - delta * l) * p * n + d2n;
    }
    T predatorFunctionTaylored(T n, T p, T d2p) const
    {
        return n * p - delta * p + d * d2p;
    }
 */
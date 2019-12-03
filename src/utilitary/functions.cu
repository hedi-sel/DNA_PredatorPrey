#include <constants.hpp>
#include <utilitary/functions.h>
#include <stdio.h>

__device__ double devLaplacien(const double *pos, const State<double> &x)
{
    return (-2 * pos[0] + x(pos, 1) + x(pos, -1)) / (dh * dh);
}

__device__ double devPreyFunction(const double n, const double p, const double d2n)
{
    return g * n * (1 - B * g * n) - p * n - l * delta * n / (1 + p) + dn * d2n;
}
__device__ double devPredatorFunction(const double n,const double p,const double d2p)
{
    return n * p - delta * p / (1 + p) + dp * d2p;
}
/*
    double preyFunctionTaylored(double n, double p, double d2n) const
    {
        return g * n * (1 - n / K) - (1 - delta * l) * p * n + d2n;
    }
    double predatorFunctionTaylored(double n, double p, double d2p) const
    {
        return n * p - delta * p + d * d2p;
    }
 */
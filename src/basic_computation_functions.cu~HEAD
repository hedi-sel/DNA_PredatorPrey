#include "basic_computation_functions.hpp"

double laplacien(const matrix &x, int type, int position)
{
    return (-2.0 * x(type, position) + x(type, position - 1) + x(type, position + 1)) / (dh * dh);
}

double preyFunction(double n, double p, double d2n)
{
    return g * n * (1 - B * g * n) - p * n - l * delta * n / (1 + p) + d2n;
}
double predatorFunction(double n, double p, double d2p)
{
    return n * p - delta * p / (1 + p) + d * d2p;
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
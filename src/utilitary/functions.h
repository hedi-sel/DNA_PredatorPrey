#include <dataStructure/state.h>

__device__ double devLaplacien(const double *, const State<double> &);
__device__ double devPreyFunction(const double n, const double p, const double d2n);
__device__ double devPredatorFunction(const double n, const double p, const double d2p);
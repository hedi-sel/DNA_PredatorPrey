#include "predator_prey_systems.hpp"
#include "cudaComputer.hpp"
#include "boost/multi_array.hpp"

void prey_predator_system_gpu::operator()(const matrix &x, matrix &dxdt, double t) const
{
    compute(x.data().begin(), &x.data().begin()[x.size2()], dxdt.data().begin(), x.size2());
}

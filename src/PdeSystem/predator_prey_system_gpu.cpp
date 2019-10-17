#include "predator_prey_systems.hpp"
#include <basic_computation_functions.hpp>
#include "cudaComputer.hpp"
#include "boost/multi_array.hpp"

void prey_predator_system_gpu::operator()(const matrix &x, matrix &dxdt, double t) const
{
    compute(x.data().begin(), &x.data().begin()[x.size2()], dxdt.data().begin(), x.size2());
    //dxdt.data() = boost::numeric::ublas::unbounded_array<double>(x.size1() * x.size2(), b[0]);
    std::cout << t << " " << x(0, 10) << " " << x(1, 10) << " ";
    std::cout << x.data().begin()[10] << " " << x.data().begin()[10 + x.size2()] << " " << dxdt.data().size() << std::endl; /* 
    for (size_t i = 0; i < x.size1(); i++)
        for (size_t j = 0; j < x.size2(); j++)
            dxdt(i, j) = b[j + i * x.size2()]; */
}

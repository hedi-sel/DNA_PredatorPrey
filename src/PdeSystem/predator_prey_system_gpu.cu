#include "predator_prey_system_gpu.hpp"
#include "../basic_computation_functions.hpp"

void prey_predator_system_gpu::operator()(const matrix &x, matrix &dxdt, double /* t */) const{
	size_t size1 = x.size1(), size2 = x.size2();
	for (size_t j = 1; j < size2 - 1; ++j)
	{
		dxdt(0, j) = preyFunction(x(0, j), x(1, j), laplacien(x, 0, j));
		dxdt(1, j) = predatorFunction(x(0, j), x(1, j), laplacien(x, 1, j));
	}

	for (size_t i = 0; i < x.size1(); ++i)
		dxdt(i, 0) = dxdt(i, x.size2() - 1) = 0.0;
}

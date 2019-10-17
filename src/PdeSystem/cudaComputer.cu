#include "cudaComputer.hpp"
#include <basic_computation_functions.hpp>
#include <array>
void compute(const double* pred, const double* prey, double* diff, int size){
    //double* diff = new double [2*size]; 
    for (size_t j = 1; j < size - 1; ++j)
    {
        diff[j] = preyFunction(prey[j], pred[j], laplacien(&prey[j]));
        diff[j + size] = predatorFunction(prey[j], pred[j], laplacien(&pred[j]));
    }

    for (size_t i = 0; i < 2; ++i)
        diff[size * i] = diff[size * (i+1) - 1] = 0.0;
    
}
// #include <pybind11/pybind11.h>

#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>
#include "launcher.h"
#include "cuda_runtime.h"

// int add(int i, int j)
// {
//     return i + j;
// }

// PYBIND11_MODULE(dna, m)
// {
//     m.doc() = "pybind11 example plugin";
//     m.def("add", &add, "addition");
// }

int main(int argc, char **argv)
{
    launch();
    return 0;
}
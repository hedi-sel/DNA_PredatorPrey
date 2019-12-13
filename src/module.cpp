#include <pybind11/pybind11.h>
#include "launcher.h"

namespace py = pybind11;

// int add(int i, int j)
// {
//     return i + j;
// }

PYBIND11_MODULE(dna, m)
{
     m.doc() = "pybind11 example plugin";
     m.def("launch", &launch, "addition");
}

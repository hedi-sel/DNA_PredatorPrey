#include <pybind11/pybind11.h>
#include "launcher.h"

namespace py = pybind11;

int Launch()
{
     launch();
}

PYBIND11_MODULE(dna, m)
{
     m.def("launch", &Launch);
}

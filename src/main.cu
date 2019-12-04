#include <iostream>
#include <string>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/timer/timer.hpp>

#include "constants.hpp"
#include "PDESystemCUDA/iterator_system.hpp"
#include "utilitary/initialConditionsBuilder.hpp"

using boost::timer::cpu_timer;

State<T> x(nSpecies, sampleSizeX, sampleSizeY, true);
State<T> *x_dev;

void initialization()
{
    int centerRabb = centerRabbRaw / dx;
    int widthRabb = widthRabbRaw / dx;
    int centerPred = centerPredRaw / dx;
    int widthPred = widthPredRaw / dx;
    gaussianMaker(x, 0, sampleSizeX, sampleSizeY, maxRabb, centerRabb, widthRabb);
    gaussianMaker(x, 1, sampleSizeX, sampleSizeY, maxPred, centerPred, widthPred);
}

//Will run the program twice (once for CPU, once for GPU) and output the runtime for each
void CpuGpuCompare()
{
    std::cout << "Setup done, starting computation" << std::endl;

    cpu_timer timer_gpu;

    Iterator_system iterator(x, t0, printPeriod);
    iterator.Iterate(dt, tmax);

    T run_time_gpu = static_cast<T>(timer_gpu.elapsed().wall) * 1.0e-9;

    std::cout << "Ended computation in: " << run_time_gpu << "s" << std::endl;

    std::cout << "Results saved in : \n"
              << iterator.outputPath << std::endl;

    std::ofstream fout("dataName");
    fout << iterator.dataName;
}

//Will output the runtime in seconds in the file performance/*dataName*
void PerformanceOriented(char arg)
{
    cpu_timer timer;

    std::string dataName = "cpu";
    if (arg == 'g')
    {
        Iterator_system iterator(x, t0, 0);
        iterator.Iterate(dt, tmax);
        dataName = iterator.dataName;
    }
    T run_time = static_cast<T>(timer.elapsed().wall) * 1.0e-9;
    std::cout /* << " -Computation Time: " */ << run_time << /* "s" <<  */ std::endl;

    std::ostringstream stream;
    stream << "performance/" << dataName;
    std::ofstream fout(stream.str(), std::ios_base::app);
    fout << run_time << std::endl;
}

int main(int argc, char **argv)
{
    initialization();

    if (argc > 1)
    {
        char arg = argv[1][0];
        if (arg == 'c' || arg == 'g')
            PerformanceOriented(arg);
        else
            PerformanceOriented('g');
    }
    else
    {
        CpuGpuCompare();
    }

    return 0;
}
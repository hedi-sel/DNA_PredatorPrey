#include <iostream>
#include <string>
#include <stdio.h>
#include <boost/filesystem.hpp>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include <boost/timer/timer.hpp>

#include "constants.hpp"
#include "PdeSystem/predator_prey_systems.hpp"
#include "PDESystemCUDA/iterator_system.hpp"
#include "PdeSystem/cudaComputer.hpp"
#include "utilitary/writeSnapshots.hpp"
#include "utilitary/initialConditionsBuilder.hpp"

using boost::timer::cpu_timer;
using namespace std;

int main(int argc, char **argv)
{

    const size_t nSpecies = 2, sampleSize = (size_t)(xLength / dh);

    //Matrix initialization
    //
    matrix x(nSpecies, sampleSize, 0.0);
    matrix y(nSpecies, sampleSize, 0.0);

    int centerRabb = centerRabbRaw / dh;
    int widthRabb = widthRabbRaw / dh;
    int centerPred = centerPredRaw / dh;
    int widthPred = widthPredRaw / dh;

    gaussianMaker(x, 0, sampleSize, maxRabb, centerRabb, widthRabb);
    gaussianMaker(y, 0, sampleSize, maxRabb, centerRabb, widthRabb);
    gaussianMaker(x, 1, sampleSize, maxPred, centerPred, widthPred);
    gaussianMaker(y, 1, sampleSize, maxPred, centerPred, widthPred);

    // Initialize Observer, for result printing

    write_snapshots snapshots;
    auto snap = [&snapshots, CpuOutputPath, dt](int n) {
        ostringstream stream;
        stream << CpuOutputPath << "/state_at_t=" << double(n) * dt << "s.dat";
        snapshots.snapshots().insert(make_pair(size_t(n), stream.str()));
    };

    // Matrix Initialization

    for (int i = 0; i <= (int)(tmax / printPeriod); i++)
        snap((int)(i * printPeriod / dt));
    observer_collection<matrix, double> obs;
    obs.observers().push_back(snapshots);

    boost::filesystem::path p(CpuOutputPath);
    boost::filesystem::directory_iterator it(p);
    for (const auto &entry : it)
        remove(entry.path());

    cout << "Setup done, starting computation" << endl;

    cpu_timer timer;
    integrate_const(runge_kutta4<matrix>(), prey_predator_system(1.2),
                    y, 0.0, tmax, dt);//, boost::ref(obs));
    double run_time = static_cast<double>(timer.elapsed().wall) * 1.0e-9;

    cpu_timer timer_custom_gpu;
    Iterator_system iterator(x.data().begin(), nSpecies, sampleSize, t0, 0);
    iterator.iterate(dt, tmax);
    double run_time_custom_gpu = static_cast<double>(timer_custom_gpu.elapsed().wall) * 1.0e-9;

    std::cout << "Ended computation in: " << endl;
    std::cout << " -Cpu Computation: " << run_time << "s" << std::endl;
    std::cout << " -GPU Computation: " << run_time_custom_gpu << "s" << std::endl;

    std::cout << "Results saved in : \n"
              << iterator.outputPath << std::endl;

    std::ofstream fout("dataName");
    fout << iterator.dataName;

    return 0;
}

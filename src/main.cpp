#include <iostream>
#include <string>
#include <stdio.h>
#include <boost/filesystem.hpp>

#include "constants.hpp"
#include "PdeSystem/predator_prey_systems.hpp"
#include "PDESystemCUDA/iterator_system.hpp"
#include "PdeSystem/cudaComputer.hpp"

#include "writeSnapshots.hpp"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using namespace std;

int main(int argc, char **argv)
{

    const size_t nSpecies = 2, sampleSize = (size_t) (xLength/dh);

    //Matrix initialization
    //
    matrix x(nSpecies, sampleSize, 0.0);
    matrix y(nSpecies, sampleSize, 0.0);

    int centerRabb = 10;
    int widthRabb = 8;
    double maxRabb = 0.6;

    int centerPred = 4;
    int widthPred = 3;
    double maxPred = 0.3;

    for (size_t j = (centerRabb - widthRabb); j < (centerRabb + widthRabb); ++j)
    {
        x(0, j) = y(0, j) = (1.0 - (j - centerRabb) * (j - centerRabb) / float(widthRabb * widthRabb)) * maxRabb;
    }
    for (size_t j = (centerPred - widthPred); j < (centerPred + widthPred); ++j)
    {
        x(1, j) = y(1, j) = (1.0 - (j - centerPred) * (j - centerPred) / float(widthRabb * widthRabb)) * maxPred;
    }

    // Initialize Observer, for result printing

    write_snapshots snapshots;
    auto snap = [&snapshots, CpuOutputPath, dt](int n) {
        ostringstream stream;
        stream << CpuOutputPath << "/state_at_t=" << double(n) * dt << "s.dat";
        snapshots.snapshots().insert(make_pair(size_t(n), stream.str()));
    };

    // Matrix Initialization

    for (int i = 0; i < (int)tmax; i++)
        snap((int)(i / dt));
    observer_collection<matrix, double> obs;
    obs.observers().push_back(snapshots);

    boost::filesystem::path P[2] = {CpuOutputPath, GpuOutputPath};
    boost::filesystem::path p2(GpuOutputPath);
    for (boost::filesystem::path p : P){
    boost::filesystem::directory_iterator it(p);
    for (const auto &entry : it)
        remove(entry.path());
    }

    cout << "Setup done, starting computation" << endl;

    cpu_timer timer;
    integrate_const(runge_kutta4<matrix>(), prey_predator_system(1.2),
                    y, 0.0, tmax, dt, boost::ref(obs));
    double run_time = static_cast<double>(timer.elapsed().wall) * 1.0e-9;

    cpu_timer timer_custom_gpu;
    Iterator_system iterator(x.data().begin(), nSpecies, sampleSize, t0, true);
    while (iterator.t < tmax)
    {
        iterator.iterate(dt);
    }
    double run_time_custom_gpu = static_cast<double>(timer_custom_gpu.elapsed().wall) * 1.0e-9;
 
    cout << "Ended computation in: " << endl;
    std::cout << " -Cpu Computation: " << run_time << "s" << std::endl;
    std::cout << " -GPU Computation: " << run_time_custom_gpu << "s" << std::endl;

    return 0;
}

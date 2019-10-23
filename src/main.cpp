#include <iostream>
#include <string>
#include <stdio.h>
#include <boost/filesystem.hpp>

#include "constants.hpp"
#include "PdeSystem/predator_prey_systems.hpp"
#include "PdeSystem/cudaComputer.hpp"

#include "writeSnapshots.hpp"

#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using namespace std;

int main(int argc, char **argv)
{
    std::cout << boost::filesystem::current_path() << std::endl;
    string outputPath = "./output";
    size_t size1 = 2, size2 = 8192;
    matrix x(size1, size2, 0.0);
    matrix y(size1, size2, 0.0);

    int centerRabb = 10;
    int widthRabb = 8;
    double maxRabb = 0.6;

    int centerPred = 4;
    int widthPred = 3;
    double maxPred = 0.3;

    for (size_t j = (centerRabb - widthRabb); j < (centerRabb + widthRabb); ++j)
    {
        x(0, j) = (1.0 - (j - centerRabb) * (j - centerRabb) / float(widthRabb * widthRabb)) * maxRabb;
        y(0, j) = (1.0 - (j - centerRabb) * (j - centerRabb) / float(widthRabb * widthRabb)) * maxRabb;
    }
    for (size_t j = (centerPred - widthPred); j < (centerPred + widthPred); ++j)
    {
        x(1, j) = (1.0 - (j - centerPred) * (j - centerPred) / float(widthRabb * widthRabb)) * maxPred;
        y(1, j) = (1.0 - (j - centerPred) * (j - centerPred) / float(widthRabb * widthRabb)) * maxPred;
    }
    write_snapshots snapshots;
    auto snap = [&snapshots, &outputPath](int n) {
        ostringstream stream;
        stream << outputPath << "/data" << double(n) * dt << "s.dat";
        snapshots.snapshots().insert(make_pair(size_t(n), stream.str()));
    };
    for (int i = 0; i < 11; i++)
        snap(i * 10);
    observer_collection<matrix, double> obs;
    obs.observers().push_back(snapshots);

    boost::filesystem::path p(outputPath);
    boost::filesystem::directory_iterator it(p);
    for (const auto &entry : it)
        remove(entry.path());

    cout << "Setup done, starting computation" << endl;
    cpu_timer timer_gpu;

    integrate_const(runge_kutta4<matrix>(), prey_predator_system_gpu(2.4), x, 0.0, 10.0, dt, boost::ref(obs));
    double run_time_gpu = static_cast<double>(timer_gpu.elapsed().wall) * 1.0e-9;

    cpu_timer timer;
    integrate_const(runge_kutta4<matrix>(), prey_predator_system(1.2),
                    y, 0.0, 10.0, dt);
    double run_time = static_cast<double>(timer.elapsed().wall) * 1.0e-9;

    cout << "Ended computation in: " << endl;
    std::cout << " -Single thread: " << run_time << "s" << std::endl;
    std::cout << " -Multi thread("
              << "): " << run_time_gpu << "s" << std::endl;

    return 0;
}

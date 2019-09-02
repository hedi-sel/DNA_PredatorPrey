#include <iostream>
#include <string>
#include <stdio.h>
#include <boost/filesystem.hpp>

/* #include "PdeSystem/predator_prey_system.hpp" */
#include "PdeSystem/predator_prey_system_gpu.hpp"
#include "writeSnapshots.hpp"

#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using namespace std;

int main(int argc, char **argv)
{
    size_t size1 = 2, size2 = 1024;
    matrix x(size1, size2, 0.0);

    int centerRabb = 10;
    int widthRabb = 8;
    double maxRabb = 0.6;

    int centerPred = 4;
    int widthPred = 3;
    double maxPred = 0.3;

    for (size_t j = (centerRabb - widthRabb); j < (centerRabb + widthRabb); ++j)
        x(0, j) = (1.0 - (j - centerRabb) * (j - centerRabb) / float(widthRabb * widthRabb)) * maxRabb;
    for (size_t j = (centerPred - widthPred); j < (centerPred + widthPred); ++j)
        x(1, j) = (1.0 - (j - centerPred) * (j - centerPred) / float(widthRabb * widthRabb)) * maxPred;

    write_snapshots snapshots;
    auto snap = [&snapshots](int n) {
        ostringstream stream;
        stream << "output/pop_" << double(n)*dt << "s.dat";
        snapshots.snapshots().insert(make_pair(size_t(n), stream.str()));
    };
    snap(0);
    for (int i = 1; i < 101; i++)
        snap(i * 100);
    observer_collection<matrix, double> obs;
    obs.observers().push_back(snapshots);

    boost::filesystem::path p("output/");
    boost::filesystem::directory_iterator it(p);
    for (const auto & entry : it)
        remove(entry.path());

    cout << "Setup done, starting computation" << endl;

    cpu_timer timer_gpu;
    int chunk_size = size2/omp_get_max_threads();
    omp_set_schedule( omp_sched_static , chunk_size );
    integrate_adaptive(make_controlled(1E-6, 1E-6, runge_kutta_dopri5<matrix>()), prey_predator_system_gpu(1.2),
                       x, 0.0, 2000.0, dt);
    double run_time_gpu = static_cast<double>(timer_gpu.elapsed().wall) * 1.0e-9;


    /* integrate_n_steps(runge_kutta_dopri5<state_type>(), prey_predator_system(1.2),
                      x, 0.0, dt, 10000, boost::ref(obs)); */

    cpu_timer timer;
    integrate_adaptive(make_controlled(1E-6, 1E-6, runge_kutta_dopri5<matrix>()), prey_predator_system(1.2),
                       x, 0.0, 2000.0, dt, boost::ref(obs));
    double run_time = static_cast<double>(timer.elapsed().wall) * 1.0e-9;
    
    cout << "Ended computation in: " << endl;
    std::cout << " -Single thread: " << run_time << "s" << std::endl;
    std::cout << " -Multi thread(" << omp_get_max_threads() << "): " << run_time_gpu << "s" << std::endl;

    return 0;
}

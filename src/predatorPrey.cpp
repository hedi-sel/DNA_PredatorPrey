#include <iostream>

#include "predator_prey_system.hpp"
#include "predator_prey_system_gpu.hpp"
#include "writeSnapshots.hpp"

#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include <boost/timer/timer.hpp>

using boost::timer::cpu_timer;
using namespace std;

int main(int argc, char **argv)
{
    size_t size1 = 2, size2 = 128;
    state_type x(size1, size2, 0.0);

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
        stream << "output/lat_" << n << ".dat";
        snapshots.snapshots().insert(make_pair(size_t(n), stream.str()));
    };
    snap(0);
    for (int i = 1; i < 22; i++)
        snap(i * 20);
    observer_collection<state_type, double> obs;
    obs.observers().push_back(snapshots);

    cout << "Setup done, starting computation" << endl;

    cpu_timer timer;

    integrate_adaptive(make_controlled(1E-6, 1E-6, runge_kutta_dopri5<state_type>()), prey_predator_system(1.2),
                       x, 0.0, 100.0 + dt, dt, boost::ref(obs));

    /* integrate_n_steps(runge_kutta4<state_type, openmp_range_algebra>(), prey_predator_system_gpu(1.2),
                      x, 0.0, 0.01, 100); */

    double run_time = static_cast<double>(timer.elapsed().wall) * 1.0e-9;

    cout << "Ended computation in " << run_time << " seconds" << endl;
    return 0;
}

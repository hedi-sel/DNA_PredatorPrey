#include <iostream>
#include <vector>
#include <boost/random.hpp>
#include <boost/bind.hpp>
#include <boost/timer/timer.hpp>
#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>

using namespace std;
using namespace boost::numeric::odeint;
using boost::timer::cpu_timer;
using boost::math::double_constants::pi;

typedef std::vector< double > state_type;

struct phase_chain
{
    phase_chain( double gamma = 0.5 )
    : m_gamma( gamma ) { }

    void operator()( const state_type &x , state_type &dxdt , double /* t */ ) const
    {
        const size_t N = x.size();
#pragma omp parallel for schedule(runtime)
        for(size_t i = 1 ; i < N-1 ; ++i)
        {
            dxdt[i] = coupling_func( x[i+1] - x[i] ) +
                      coupling_func( x[i-1] - x[i] );
        }
        dxdt[0  ] = coupling_func( x[1  ] - x[0  ] );
        dxdt[N-1] = coupling_func( x[N-2] - x[N-1] );
    }

    double coupling_func( double x ) const
    {
        return sin( x ) - m_gamma * ( 1.0 - cos( x ) );
    }

    double m_gamma;
};


int main( int argc , char **argv )
{
    size_t N = 131101;
    state_type x( N );
    boost::random::uniform_real_distribution<double> distribution( 0.0 , 2.0*pi );
    boost::random::mt19937 engine( 0 );
    generate( x.begin() , x.end() , boost::bind( distribution , engine ) );

    typedef runge_kutta4<
                      state_type , double ,
                      state_type , double ,
                      openmp_range_algebra
                    > stepper_type;

    int chunk_size = N/omp_get_max_threads();
    omp_set_schedule( omp_sched_static , chunk_size );
    
    cpu_timer timer;
    integrate_n_steps(runge_kutta4<state_type, double, state_type, double>(), phase_chain(1.2),
                      x, 0.0, 0.01, 100);
    double run_time = static_cast<double>(timer.elapsed().wall) * 1.0e-9;

    cpu_timer timer_gpu;
    integrate_n_steps(stepper_type(), phase_chain(1.2), x, 0.0, 0.01, 100);
    double run_time_gpu = static_cast<double>(timer_gpu.elapsed().wall) * 1.0e-9;

    std::cout << "Single thread:" << run_time << "s" << std::endl;
    std::cout << "Multi thread:" << run_time_gpu << "s" << std::endl;

    return 0;
}
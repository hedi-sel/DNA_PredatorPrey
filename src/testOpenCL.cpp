#include <omp.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/openmp/openmp.hpp>
#include <boost/random.hpp>

typedef std::vector<double> state_type;

size_t N = 131101;
state_type x(N);

boost::random::uniform_real_distribution<double> distribution(0.0, 2.0 * pi);
boost::random::mt19937 engine(0);

boost::random_number_generator(x.begin(), x.end(), boost::bind(distribution, engine));

typedef runge_kutta4<	state_type, double,	state_type, double,	openmp_range_algebra>	stepper_type;
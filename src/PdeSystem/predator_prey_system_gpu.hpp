#include <iostream>
#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;
typedef boost::numeric::ublas::matrix<double> matrix;

class prey_predator_system_gpu
{
public:
	prey_predator_system_gpu(double gamma = 0.5)
			: m_gamma(gamma) {}

	void operator()(const matrix &x, matrix &dxdt, double /* t */) const;

private:
	double m_gamma;
};
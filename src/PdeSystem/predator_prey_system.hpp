#include <iostream>
#include <boost/numeric/odeint.hpp>
using namespace boost::numeric::odeint;
typedef std::vector<double> matrix;

class prey_predator_system
{
public:
    prey_predator_system(double gamma = 0.5)
            : m_gamma(gamma) {}

    void operator()(const matrix &x, matrix &dxdt, double) const;

private:
    double m_gamma;
};
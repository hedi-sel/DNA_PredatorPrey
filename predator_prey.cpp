#include <iostream>
#include <fstream>
#include <boost/numeric/odeint.hpp>
#include <boost/array.hpp>
//#include <boost/numeric/odeint/external/vexcl/vexcl.hpp>
//#include <boost/numeric/odeint/external/thrust/thrust.hpp>

using namespace boost::numeric::odeint;

const double a = 10.0;
const double b = 8.0;

typedef boost::array< double , 2 > state_type;
//typedef thrust::host_vector< value_type > state_type;

std::ofstream dataFile;

void equation( const state_type &x , state_type &dxdt , double t )
{
    dxdt[0] = a * ( x[1] - x[0] );
    dxdt[1] = -b + x[0] * x[1];
}

void writeFunction( const state_type &x , const double t )
{
    dataFile << t << '\t' << x[0] << '\t' << x[1] << std::endl;
}

int main(int argc, char **argv)
{
    dataFile.open ("OutputData.dat");
    state_type x = {{ 10.0 , 1.0}}; // initial conditions
    integrate( equation , x , 0.0 , 25.0 , 0.1 , writeFunction );
    dataFile.close();
    return 0;
}

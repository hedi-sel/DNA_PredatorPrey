/*
 * prey_predator_system.cpp
 *
 * This example show how one can use matrices as state types in odeint.
 *
 * Copyright 2011-2012 Karsten Ahnert
 * Copyright 2011-2013 Mario Mulansky
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 */
// ready

#include <iostream>
#include <map>
#include <string>
#include <fstream>

#ifndef M_PI //not there on windows
#define M_PI 3.1415927 //...
#endif

#include <boost/numeric/odeint.hpp>

using namespace std;
using namespace boost::numeric::odeint;

//[ prey_predator_system_definition
typedef boost::numeric::ublas::matrix< double > state_type;

const double k1 = 0.003;
const double k2 = 0.004;
const double kn = 0.01;
const double kp = 0.004;
const double b = 6.0e-5;
const double Kmp = 34.0;
const double pol = 1.7;
const double exo = 25.0;
const double G = 140.0;
const double Dp = 2000.0;
const double tc = 1 / (k2 * pol * Kmp);
const double g = k1 * G / (k2 * Kmp);
const double B = b * k2 * Kmp * Kmp / k1;
const double l = kn / kp;
const double delta = (exo / pol) * (kp / k2 / Kmp);
const double A = g - l * delta;
const double K = (g - l * delta) / (B * g * g);
const double d = 1;//TODO
const double dh = 1e-1;//TODO

struct prey_predator_system
{

    prey_predator_system( double gamma = 0.5 )
    : m_gamma( gamma ) { }

    void operator()( const state_type &x , state_type &dxdt , double /* t */ ) const
    {
        size_t size1 = x.size1() , size2 = x.size2();
        for (size_t j = 1; j < size2 - 1; ++j)
        {
            dxdt(0, j) = preyFunction(x(0, j), x(1, j), laplacien(x, 0, j));
            dxdt(1, j) = predatorFunction(x(0, j), x(1, j), laplacien(x, 1, j));
        }

        for( size_t i=0 ; i<x.size1() ; ++i ) dxdt( i , 0 ) = dxdt( i , x.size2() -1 ) = 0.0;
    }

    double laplacien(const state_type &x, int type, int position) const
    {
        return (2.0*x(type, position) - x(type, position-1) - x(type, position+1.0));
    }

    double preyFunction( double n, double p, double Dn ) const
    {
        return A*n*(1-n/K)-(1-delta*l)*p*n+Dn;
    }
    double predatorFunction( double n, double p, double Dp ) const
    {
        return n*p - delta*p + d*Dp;
    }

    double m_gamma;
};

class write_snapshots
{
public:

    typedef std::map< size_t , std::string > map_type;

    write_snapshots( void ) : m_count( 0 ) { }

    void operator()( const state_type &x , double t )
    {
        map< size_t , string >::const_iterator it = m_snapshots.find( m_count );
        if( it != m_snapshots.end() )
        {
            ofstream fout( it->second.c_str() );
            fout << x.size1() << "\t" << x.size2() << "\n";
            for( size_t i=0 ; i<x.size1() ; ++i )
            {
                for( size_t j=0 ; j<x.size2() ; ++j )
                {
                    fout << i << "\t" << j << "\t" << x( i , j ) << "\n";
                }
            }
        }
        ++m_count;
    }

    map_type& snapshots( void ) { return m_snapshots; }
    const map_type& snapshots( void ) const { return m_snapshots; }

private:

    size_t m_count;
    map_type m_snapshots;
};


int main( int argc , char **argv )
{
    size_t size1 = 2 , size2 = 128;
    state_type x( size1 , size2 , 0.0 );

    for( size_t j=(size2/2-10) ; j<(size2/2+10) ; ++j )
        x( 0 , j ) = (1.0 - (j - size2/2) * (j - size2/2)/100.0)/1000.0;

    write_snapshots snapshots;
    auto snap = [&snapshots](int n) {
        ostringstream stream;
        stream << "data/lat_" << n << ".dat";
        snapshots.snapshots().insert(make_pair(size_t(n), stream.str()));
    };
    snap(0);
    for (int i = 1; i<5; i++) snap(i);
    observer_collection<state_type, double> obs;
    obs.observers().push_back( snapshots );

    cout << "Setup done, starting computation" << endl;

    integrate_const(runge_kutta4<state_type>(), prey_predator_system(1.2),
                    x, 0.0, 10.0, 0.01, boost::ref(obs));

    // controlled steppers work only after ublas bugfix
    //integrate_const( make_dense_output< runge_kutta_dopri5< state_type > >( 1E-6 , 1E-6 ) , prey_predator_system( 1.2 ) ,
    //        x , 0.0 , 1001.0 , 0.1 , boost::ref( obs ) );

    cout << "Fini" << endl;

    return 0;
}

#include <iostream>
#include <boost/numeric/odeint.hpp>

using namespace boost::numeric::odeint;

typedef boost::numeric::ublas::matrix<double> matrix;


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
const double d = 1.0;		//TODO
const double dh = 10;		//TODO
const double dt = 0.01; //TODO
const double C = 1.3;
const double testvql = -(C * C + delta) * B * g * g + g;

double laplacien(const matrix &x, int type, int position)
{
	return (-2.0 * x(type, position) + x(type, position - 1) + x(type, position + 1)) / (dh * dh);
}

double preyFunction(double n, double p, double d2n)
{
	return g * n * (1 - B * g * n) - p * n - l * delta * n / (1 + p) + d2n;
}
double predatorFunction(double n, double p, double d2p)
{
	return n * p - delta * p / (1 + p) + d * d2p;
}
/*
	double preyFunctionTaylored(double n, double p, double d2n) const
	{
		return g * n * (1 - n / K) - (1 - delta * l) * p * n + d2n;
	}
	double predatorFunctionTaylored(double n, double p, double d2p) const
	{
		return n * p - delta * p + d * d2p;
	}
 */
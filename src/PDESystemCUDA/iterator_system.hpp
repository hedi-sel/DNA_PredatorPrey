#include <iostream>

typedef void (*Stepper)(double *, double *, int , int , double , double);

class Iterator_system
{
public:
    int nSpecies;
    int sampleSize;
    double t;
    bool doPrint;
    Iterator_system(double *, int, int, double, bool);
    ~Iterator_system();
    void iterate(double);
    void printer(double);

private:
    double printPeriod = 1.0;
    double nextPrint = 0.0;
    double *x;
    double *dxdt;
    Stepper stepper;
};
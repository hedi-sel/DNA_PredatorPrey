#include <iostream>

class iterator_system
{
public:
    int nSpecies;
    int sampleSize;
    double t;
    bool doPrint;
    iterator_system(double *, int, int, double, bool);
    ~iterator_system();
    void iterate(double);
    void printer(double);

private:
    double printPeriod = 1.0;
    double nextPrint = 0.0;
    double *x;
    double *dxdt;
};
#include <iostream>

class prey_predator_iterator
{
public:
    int nSpecies;
    int sampleSize;
    double t;
    bool doPrint;
    prey_predator_iterator(double *, int, int, double, bool);
    ~prey_predator_iterator();
    void iterate(double);
    void printer(double);

private:
    double printPeriod = 1.0;
    double nextPrint = 0.0;
    double *x;
    double *dxdt;
};
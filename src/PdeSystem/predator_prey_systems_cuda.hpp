#include <iostream>

class prey_predator_iterator
{
public:
    int im;
    int jm;
    double t=0.0;
    bool doPrint;
    prey_predator_iterator(double *, int, int, bool);
    ~prey_predator_iterator();
    void iterate(double);
    void printer(double);

private:
    double printPeriod = 1.0;
    double nextPrint = 0.0;
    double *x;
    double *dxdt;
};
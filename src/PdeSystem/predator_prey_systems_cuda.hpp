#include <iostream>

class prey_predator_iterator
{
public:
    int im;
    int jm;
    int stepCount = 0;
    bool doPrint;
    double *x;
    double *dxdt;
    double snapPeriod;
    prey_predator_iterator(double *, int, int, bool);
    ~prey_predator_iterator();
    void iterate(double, double);
    void printer(double);
};
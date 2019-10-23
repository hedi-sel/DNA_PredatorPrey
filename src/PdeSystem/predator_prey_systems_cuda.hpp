#include <iostream>

class prey_predator_iterator
{
public:
    int im;
    int jm;
    double *x;
    double snapPeriod;
    prey_predator_iterator(double *, int, int, double);
    ~prey_predator_iterator();
    void iterate(double, double);
    void printer(double, double, double);
};
#include <iostream>
#include <string>

typedef void (*Stepper)(double *, double *, int , int , double , double);

class Iterator_system
{
public:
    int nSpecies;
    int sampleSize;
    double t;
    bool doPrint;
    std::string outputPath;
    std::string dataName;

    Iterator_system(double *, int, int, double, double);
    ~Iterator_system();
    void iterate(double);
    void iterate(double, double);
    void iterate(double, int);
    void printer();
    void printer(double);

private:
    double printPeriod = 1.0;
    double nextPrint = 0.0;
    double *x;
    double *dxdt;
    Stepper stepper;
};
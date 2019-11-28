#include <iostream>
#include <string>
#include <dataStructure/state.h>

typedef void (*Stepper)(State<double>&, double, double);

class Iterator_system
{
public:
    double t;
    bool doPrint;
    std::string outputPath;
    std::string dataName;

    Iterator_system(State<double>&, double, double);
    ~Iterator_system();
    void Iterate(double);
    void Iterate(double, double);
    void Iterate(double, int);
    void Print();
    void Print(double);

private:
    double printPeriod = 1.0;
    double nextPrint = 0.0;
    State<double>& state;
    Stepper stepper;
};
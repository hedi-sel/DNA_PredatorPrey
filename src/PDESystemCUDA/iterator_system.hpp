#include <iostream>
#include <string>
#include <dataStructure/state.h>
#include <constants.hpp>

typedef void (*Stepper)(State<T> &, T, T);
// typedef void (*Derivator)(State<T> &, T, dim3);

class Iterator_system
{
public:
    T t;
    bool doPrint;
    std::string outputPath;
    std::string dataName;

    Iterator_system(State<T>&, T, T);
    ~Iterator_system();
    void Iterate(T);
    void Iterate(T, T);
    void Iterate(T, int);
    void Print();
    void Print(T);

private:
    T printPeriod = 1.0;
    T nextPrint = 0.0;
    State<T>& state;
    Stepper stepper;
};
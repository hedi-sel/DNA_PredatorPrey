#include <iostream>
#include <stdio.h>
#include <math.h>
#include <dataStructure/state.h>

void gaussianMaker(State<double> &m, int species, int sampleSize, double max, int center, int width)
{
    auto function = [max, center, width](int j) {
        return max * exp(-pow(j - center,2) / double(width * width));
    };
    for (int j = 0; j < sampleSize; ++j)
    {
        m(species, j) = function(j);
/* 
        if (species == 0 && j >3658 && j< 3661)
            printf("%i %f \n",j, - pow(j - center, 2) / double(width * width)); */
    }
}
#include <iostream>
#include <math.h>
#include <dataStructure/state.h>

void gaussianMaker(State<double> &m, int species, int sampleSize, double max, int center, int width)
{
    auto function = [max, center, width](int j) {
        return max * exp(-(j - center) * (j - center) / double(width * width));
    };

    for (int j = 0; j < sampleSize; ++j)
    {
        m(species, j) = function(j);
    }
}
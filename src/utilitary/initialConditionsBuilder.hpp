#include <iostream>
#include <boost/numeric/odeint.hpp>
typedef boost::numeric::ublas::matrix<double> matrix;

void gaussianMaker(matrix &matrix, int species, int sampleSize, double max, int center, int width)
{
    auto function = [max, center, width](int j) {
        return max * exp(-(j - center) * (j - center) / double(width * width));
    };

    for (int j = 0; j < sampleSize; ++j)
    {
        matrix(species, j) = function(j);
    }
}
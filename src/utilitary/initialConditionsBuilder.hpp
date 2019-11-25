#include <boost/numeric/odeint.hpp>
typedef boost::numeric::ublas::matrix<double> matrix;

void gaussianMaker(matrix &matrix, int species, int sampleSize, double max, int center, int width)
{
    auto function = [&](size_t j) {
        return max * exp(-(j - center) * (j - center) / double(width * width));
    };

    for (size_t j = 0; j < sampleSize - 1; ++j)
    {
        matrix(species, j) = function(j);
    }
}
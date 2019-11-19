#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <string>
#include <stdio.h>
#include <assert.h>

#include "iterator_system.hpp"
#include <constants.hpp>
#include <functions.h>

#include "runge_kutta_4_stepper.cu"

#define gpuErrchk(ans)                        \
{                                         \
    gpuAssert((ans), __FILE__, __LINE__); \
}

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}


iterator_system::iterator_system(double *x, int nSpecies, int sampleSize, double t0, bool print)
{
    this->nSpecies = nSpecies;
    this->sampleSize = sampleSize;
    this->doPrint = print;
    this->t = t0;
    gpuErrchk(cudaMalloc(&this->x, nSpecies * sampleSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&this->dxdt, nSpecies * sampleSize * sizeof(double)));
    gpuErrchk(cudaMemcpy(this->x, x, nSpecies * sampleSize * sizeof(double), cudaMemcpyHostToDevice));

    if (doPrint)
    {
        printer(0.0);
        nextPrint += printPeriod;
    }
}
iterator_system::~iterator_system()
{
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaFree(dxdt));
};

void iterator_system::iterate(double dt)
{
    t += dt;
    dim3 threadsPerBlock(32, 32);
    int numBlocks = (nSpecies * sampleSize + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    assert(numBlocks * 32 * 32 > nSpecies * sampleSize);
    rungeKutta4Stepper<<<numBlocks, threadsPerBlock>>>(x, dxdt, nSpecies, sampleSize, t, dt);
    if (doPrint && t >= nextPrint)
    {
        printer(t);
        nextPrint += printPeriod;
    }
    //TODO check why x doesn't change
}

void iterator_system::printer(double t)
{
    double *xHost = new double[nSpecies * sampleSize];
    cudaMemcpy(xHost, x, nSpecies * sampleSize * sizeof(double), cudaMemcpyDeviceToHost);

    std::ostringstream stream;
    stream << GpuOutputPath << "/state_at_t=" << t << "s.dat";
    std::ofstream fout(stream.str());
    fout << nSpecies << "\t" << sampleSize << "\n";
    for (size_t i = 0; i < nSpecies; ++i){
        for (size_t j = 0; j < sampleSize; ++j)
        {
            fout << i << "\t" << j << "\t" << xHost[i*sampleSize + j] << "\n";
        }
    }
}
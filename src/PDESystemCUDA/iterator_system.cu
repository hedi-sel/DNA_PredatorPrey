#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <string>
#include <stdio.h>
#include <assert.h>

#include "runge_kutta_4_stepper.hpp"
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

Iterator_system::Iterator_system(double *x, int nSpecies, int sampleSize, double t0, double print)
{
    this->nSpecies = nSpecies;
    this->sampleSize = sampleSize;
    this->doPrint = print > 0;
    this->printPeriod = print;
    this->t = t0;
    gpuErrchk(cudaMalloc(&this->x, nSpecies * sampleSize * sizeof(double)));
    gpuErrchk(cudaMalloc(&this->dxdt, nSpecies * sampleSize * sizeof(double)));
    gpuErrchk(cudaMemcpy(this->x, x, nSpecies * sampleSize * sizeof(double), cudaMemcpyHostToDevice));

    this->stepper = rungeKutta4Stepper;

    if (doPrint)
    {
        printer();
        printer(0.0);
        nextPrint += printPeriod;
    }
}
Iterator_system::~Iterator_system()
{
    gpuErrchk(cudaFree(x));
    gpuErrchk(cudaFree(dxdt));
};

void Iterator_system::iterate(double dt)
{
    t += dt;
    dim3 threadsPerBlock(32, 32);
    int numBlocks = (nSpecies * sampleSize + threadsPerBlock.x * threadsPerBlock.y - 1) / (threadsPerBlock.x * threadsPerBlock.y);
    assert(numBlocks * 32 * 32 > nSpecies * sampleSize);
    stepper(x, dxdt, nSpecies, sampleSize, t, dt);
    //rungeKutta4Stepper<<<numBlocks, threadsPerBlock>>>(x, dxdt, nSpecies, sampleSize, t, dt);
    if (doPrint && t >= nextPrint - dt / 2.)
    {
        printer(t);
        nextPrint += printPeriod;
    }
}

void Iterator_system::iterate(double dt, double tmax)
{
    int n_points = 50;
    auto printProgress = [n_points](double start, double end, double current) {
        int current_point = (int)((n_points + 1) * (current - start) / (end - start));
        std::cout << "\r [";
        for (int i = 0; i < n_points; i++)
        {
            if (i <= current_point)
                std::cout << "#";
            else
                std::cout << ".";
        };
        std::cout << "]";
    };
    double start = this->t;
    int printPeriod = (int)(tmax - start) / (dt * n_points);
    int timeSinceLastPrint = 0;
    while (this->t < tmax - dt / 2.0)
    {
        iterate(dt);
        timeSinceLastPrint++;
        if (timeSinceLastPrint > printPeriod)
        {
            timeSinceLastPrint = 0;
            printProgress(start, tmax, t);
        };
    };
    std::cout << std::endl;
}

void Iterator_system::iterate(double dt, int n_steps)
{
    iterate(dt, t + dt * n_steps);
}

void Iterator_system::printer()
{
    std::ostringstream stream;
    stream << "x=" << xLength / 1000.0 << "mm_dh=" << dh << "µm_t=" << tmax << "s_dt=" << dt * 1000.0 << "ms";
    dataName = stream.str();

    std::ostringstream stream2;
    stream2 << GpuOutputPath << "/" << dataName;
    outputPath = stream2.str();
    if (!boost::filesystem::exists(outputPath))
        boost::filesystem::create_directory(outputPath);
    else
    {
        boost::filesystem::path p(outputPath);
        boost::filesystem::directory_iterator it(p);
        for (const auto &entry : it)
            remove(entry.path());
    }
}

void Iterator_system::printer(double t)
{
    double *xHost = new double[nSpecies * sampleSize];
    cudaMemcpy(xHost, x, nSpecies * sampleSize * sizeof(double), cudaMemcpyDeviceToHost);
    std::ostringstream stream;
    stream << outputPath << "/state_at_t=" << t << "s.dat";
    std::ofstream fout(stream.str());
    fout << nSpecies << "\t" << sampleSize << "\n";
    for (size_t i = 0; i < nSpecies; ++i)
    {
        for (size_t j = 0; j < sampleSize; ++j)
        {
            fout << i << "\t" << j << "\t" << xHost[i * sampleSize + j] << "\n";
        }
    }
}

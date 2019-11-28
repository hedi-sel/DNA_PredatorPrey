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
#include <utilitary/functions.h>

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

Iterator_system::Iterator_system(State<double> &h_state, double t0, double print)
    : state(h_state)
{
    this->doPrint = print > 0;
    this->printPeriod = print;
    this->t = t0;
    //gpuErrchk(this.state = h_state.GetDeviceCopy());

    this->stepper = rungeKutta4Stepper;

    std::ostringstream stream;
    stream << "x=" << xLength / 1000.0 << "mm_dh=" << dh << "µm_t=" << tmax << "s_dt=" << dt * 1000.0 << "ms";
    dataName = stream.str();

    if (doPrint)
    {
        Print();
        Print(0.0);
        nextPrint += printPeriod;
    }
}

void Iterator_system::Iterate(double dt)
{
    t += dt;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    int numBlocks = (state.GetSize() - 1) / (threadsPerBlock.x * threadsPerBlock.y) + 1;
    assert(numBlocks * BLOCK_SIZE * BLOCK_SIZE > state.GetSize());
    stepper(state, t, dt);
    //rungeKutta4Stepper<<<numBlocks, threadsPerBlock>>>(x, dxdt, nSpecies, sampleSize, t, dt);
    if (doPrint && t >= nextPrint - dt / 2.)
    {
        Print(t);
        nextPrint += printPeriod;
    }
}

void Iterator_system::Iterate(double dt, double tmax)
{
    bool printendl = false;
    int n_points = 50;
    auto printProgress = [n_points, &printendl](double start, double end, double current) {
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
        printendl = true;
    };

    double start = this->t;
    int printPeriod = (int)(tmax - start) / (dt * n_points);
    int timeSinceLastPrint = 0;
    while (this->t < tmax - dt / 2.0)
    {
        Iterate(dt);
        timeSinceLastPrint++;
        if (timeSinceLastPrint > printPeriod)
        {
            timeSinceLastPrint = 0;
            printProgress(start, tmax, t);
        };
    };
    if (printendl)
        std::cout << std::endl;
}

void Iterator_system::Iterate(double dt, int n_steps)
{
    Iterate(dt, t + dt * n_steps);
}

void Iterator_system::Print()
{
    std::ostringstream stream;
    stream << GpuOutputPath << "/" << dataName;
    outputPath = stream.str();
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

void Iterator_system::Print(double t)
{
    double *xHost = new double[state.GetSize()];
    cudaMemcpy(xHost, state.GetRawData(), state.GetSize() * sizeof(double), cudaMemcpyDeviceToHost);
    std::ostringstream stream;
    stream << outputPath << "/state_at_t=" << t << "s.dat";
    std::ofstream fout(stream.str());
    fout << state.nSpecies << "\t" << state.sampleSizeX << "\n";
    for (size_t i = 0; i < state.nSpecies; ++i)
    {
        for (size_t j = 0; j < state.sampleSizeX; ++j)
        {
            fout << i << "\t" << j << "\t" << xHost[i * state.sampleSizeX + j] << "\n";
        }
    }
}

Iterator_system::~Iterator_system()
{
    gpuErrchk(cudaFree(state.GetRawData()));
};
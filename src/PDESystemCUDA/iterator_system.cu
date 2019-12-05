#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>
#include <string>
#include <stdio.h>
#include <assert.h>

#include "iterator_system.hpp"
#include "runge_kutta_4_stepper.h"
#include <constants.hpp>
#include <utilitary/functions.h>
#include <utilitary/cudaErrorCheck.h>

Iterator_system::Iterator_system(State<T> &h_state, T t0, T print)
    : state(h_state)
{
    this->doPrint = print > 0;
    this->printPeriod = print;
    this->t = t0;

    this->stepper = rungeKutta4Stepper;

    std::ostringstream stream;
    if (is2D)
        stream << "2D_";
    stream << "x=" << xLength / 1000.0 << "mm_dh=" << dx << "µm_t=" << tmax << "s_dt=" << dt * 1000.0 << "ms";
    dataName = stream.str();

    if (doPrint)
    {
        Print();
        Print(0.0);
        nextPrint += printPeriod;
    }
}

void Iterator_system::Iterate(T dt)
{
    t += dt;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    int numBlocks = (state.GetSize() - 1) / (threadsPerBlock.x * threadsPerBlock.y) + 1;
    assert(numBlocks * BLOCK_SIZE * BLOCK_SIZE > state.GetSize());
    stepper(state, t, dt);
    if (doPrint && t >= nextPrint - dt / 2.)
    {
        Print(t);
        nextPrint += printPeriod;
    }
}

const int n_points = 50;
auto printProgress = [](T start, T end, T current) {
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

void Iterator_system::Iterate(T dt, T tmax)
{
    T start = this->t;
    int printPeriod = (int)(tmax - start) / (dt * n_points);
    int timeSinceLastPrint = 0;
    while (t < tmax - dt / 2.0)
    {
        Iterate(dt);
        timeSinceLastPrint++;
        if (timeSinceLastPrint > printPeriod)
        {
            timeSinceLastPrint = 0;
            printProgress(start, tmax, t);
        };
    };
    std::cout << std::endl;
}

void Iterator_system::Iterate(T dt, int n_steps)
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

void Iterator_system::Print(T t)
{
    State<T> xHost(state, true);
    std::ostringstream stream;
    stream << outputPath << "/state_at_t=" << t << "s.dat";
    std::ofstream fout(stream.str());
    fout << state.sampleSizeX;
    if (state.sampleSizeY > 1)
        fout << "\t" << state.sampleSizeY;
    fout << "\t" << state.nSpecies << "\n";
    for (size_t i = 0; i < state.nSpecies; ++i)
    {
        for (size_t j = 0; j < state.sampleSizeX; ++j)
        {
#if is2D
            for (size_t k = 0; k < state.sampleSizeY; ++k)
            {
                fout << j << "\t" << k << "\t" << i << "\t" << xHost(i, j, k) << "\n";
            }
#else
            fout << j << "\t" << i << "\t" << xHost(i, j) << "\n";
#endif
        }
    }
}

Iterator_system::~Iterator_system(){
    //gpuErrchk(cudaFree(state.GetRawData()));
};
#include <string>
#include <dataStructure/dim3T.h>

#pragma once

const int BLOCK_SIZE = 32;

// double or float?
typedef double T;

const T k1 = 0.003;
const T k2 = 0.004;
const T kn = 0.01;
const T kp = 0.004;
const T b = 6.0e-5;
const T Kmp = 34.0;
const T pol = 1.7;
const T exo = 25.0;
const T G = 140.0;
const T Dp = 2000.0;
const T tc = 1 / (k2 * pol * Kmp);
const T g = k1 * G / (k2 * Kmp);
const T B = b * k2 * Kmp * Kmp / k1;
const T l = kn / kp;
const T delta = (exo / pol) * (kp / k2 / Kmp);
const T A = g - l * delta;
const T K = (g - l * delta) / (B * g * g);
const T dp = 0.1;
const T dn = 1.0;
const T C = 1.3;

const int nSpecies = 2;

// METADATA
//
const T t0 = 0.0;
const T tmax = 10;   //s
const T dt = 1.0e-4; //s

const T printPeriod = 1;

const T xLength = 1.0e4;   //µm
const T yLength = xLength; //µm
const T dx = 0.1;          //µm
const T dy = 0;            //µm

// Initial values
//
const T centerRabbRaw = 5000;
const T widthRabbRaw = 10000;
const T maxRabb = 1;

const T centerPredRaw = 2000;
const T widthPredRaw = 4000;
const T maxPred = 2;

const std::string CpuOutputPath = "./outputTemoin";
const std::string GpuOutputPath = "./output";

// Consequence
//
const bool is2D = dy != 0 && yLength != 0;
const int sampleSizeX = (xLength / dx);
const int sampleSizeY = (is2D) ? (yLength / dy) : 1;
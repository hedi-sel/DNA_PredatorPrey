#include <string>
#include <dataStructure/dim3T.h>
#include <dataStructure/dim2.h>

#pragma once

const int BLOCK_SIZE = 32;

// double or float?
#define USE_DOUBLE 1
#if USE_DOUBLE
typedef double T;
#else
typedef float T;
#endif

template struct dim<T>;

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
const T tmax = 100;   //s
const T dt = 10.0e-3; //s

const T printPeriod = 1;

constexpr T xLength = 1.0e2;   //µm
constexpr T yLength = xLength; //µm
#define DY 1
constexpr T dx = 1;  //µm
constexpr T dy = dx; //µm
#define is2D !(DY == 0)
// Initial values
//
const dim<T> centerRabbRaw(xLength / 2.0, xLength / 4.0);
const T widthRabbRaw = xLength;
const T maxRabb = 1;

const dim<T> centerPredRaw(xLength / 5.0, xLength / 5.0);
const T widthPredRaw = xLength / 2.0;
const T maxPred = 2;

const std::string CpuOutputPath = "./outputTemoin";
const std::string GpuOutputPath = "./output";

// Usefull Variables
//

const dim2 sampleSize((xLength / dx),
                      (is2D) ? (yLength / dy) : 1);
// Discriete version of initial conditions
const dim2 centerRabb(centerRabbRaw.x / dx, (is2D) ? centerRabbRaw.y / dy : 0);
const int widthRabb = widthRabbRaw / dx;
const dim2 centerPred(centerPredRaw.x / dx, (is2D) ? centerPredRaw.y / dy : 0);
const int widthPred = widthPredRaw / dx;
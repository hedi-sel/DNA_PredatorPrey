#include <string>

const int BLOCK_SIZE = 32;

const double k1 = 0.003;
const double k2 = 0.004;
const double kn = 0.01;
const double kp = 0.004;
const double b = 6.0e-5;
const double Kmp = 34.0;
const double pol = 1.7;
const double exo = 25.0;
const double G = 140.0;
const double Dp = 2000.0;
const double tc = 1 / (k2 * pol * Kmp);
const double g = k1 *G / (k2 * Kmp);
const double B = b * k2 * Kmp * Kmp / k1;
const double l = kn / kp;
const double delta = (exo / pol) * (kp / k2 / Kmp);
const double A = g - l * delta;
const double K = (g - l * delta) / (B * g * g);
const double dp = 0.1;
const double dn = 1.0;
const double C = 1.3;

// METADATA
//
const double t0 = 0.0;
const double tmax = 10; //s
const double dt = 1.0e-4; //s

const double printPeriod = 1;

const double xLength = 1.0e4; //µm
const double dh = 0.1; //µm

// Initial values
//
const double centerRabbRaw = 5000;
const double widthRabbRaw = 10000;
const double maxRabb = 1;

const double centerPredRaw = 2000;
const double widthPredRaw = 4000;
const double maxPred = 2;

const std::string CpuOutputPath = "./outputTemoin";
const std::string GpuOutputPath = "./output";

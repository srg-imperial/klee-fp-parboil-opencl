#include <stdlib.h>
#include <assert.h>

#include <klee/klee.h>
#include "klee-bits.h"
#include "../base/computeQ.h"
#include "../opencl/computeMRIQ.h"

int main(int argc, char **argv) {
  int numK = 2, numX = 2;
  float *kx, *ky, *kz, *x, *y, *z;
  float *phiR, *phiI;
  float *phiMag;
  float *cpuQr, *cpuQi;
  float *gpuQr, *gpuQi;
  struct kValues *kVals;

  kx = malloc(numK * sizeof(float));
  ky = malloc(numK * sizeof(float));
  kz = malloc(numK * sizeof(float));

  x = malloc(numX * sizeof(float));
  y = malloc(numX * sizeof(float));
  z = malloc(numX * sizeof(float));

  phiR = malloc(numK * sizeof(float));
  phiI = malloc(numK * sizeof(float));

  klee_make_symbolic(kx, numK * sizeof(float), "kx");
  klee_make_symbolic(ky, numK * sizeof(float), "ky");
  klee_make_symbolic(kz, numK * sizeof(float), "kz");

  klee_make_symbolic(x, numX * sizeof(float), "x");
  klee_make_symbolic(y, numX * sizeof(float), "y");
  klee_make_symbolic(z, numX * sizeof(float), "z");

  klee_make_symbolic(phiR, numK * sizeof(float), "phiR");
  klee_make_symbolic(phiI, numK * sizeof(float), "phiI");

  createDataStructsCPU(numK, numX, &phiMag, &cpuQr, &cpuQi);

  ComputePhiMagCPU(numK, phiR, phiI, phiMag);

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }
  ComputeQCPU(numK, numX, kVals, x, y, z, cpuQr, cpuQi);

  computeMRIQ_GPU(512, 256, numK, numX, kx, ky, kz, x, y, z, phiR, phiI,
                  &gpuQr, &gpuQi);

  int same = 1;
  for (int x = 0; x < numX; ++x) {
    same &= float_bitwise_eq(cpuQr[x], gpuQr[x]);
    same &= float_bitwise_eq(cpuQi[x], gpuQi[x]);
  }
  klee_print_expr("same", same);

  assert(same);
}

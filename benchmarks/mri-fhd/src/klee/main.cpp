#include <stdlib.h>
#include <assert.h>

#include <klee/klee.h>
#include "klee-bits.h"
#include "../base/computeFH.h"
#include "../opencl/computeFH.h"

int main(int argc, char **argv) {
  int numK = 2, numX = 2;
  float *kx, *ky, *kz, *x, *y, *z;
  float *phiR, *phiI;
  float *dR, *dI;
  float *realRhoPhi, *imagRhoPhi;
  float *cpuOutR, *cpuOutI;
  float *gpuOutR, *gpuOutI;
  struct kValues *kVals;

  kx = (float *) malloc(numK * sizeof(float));
  ky = (float *) malloc(numK * sizeof(float));
  kz = (float *) malloc(numK * sizeof(float));

  x = (float *) malloc(numX * sizeof(float));
  y = (float *) malloc(numX * sizeof(float));
  z = (float *) malloc(numX * sizeof(float));

  phiR = (float *) malloc(numK * sizeof(float));
  phiI = (float *) malloc(numK * sizeof(float));

  dR = (float *) malloc(numK * sizeof(float));
  dI = (float *) malloc(numK * sizeof(float));

  klee_make_symbolic(kx, numK * sizeof(float), "kx");
  klee_make_symbolic(ky, numK * sizeof(float), "ky");
  klee_make_symbolic(kz, numK * sizeof(float), "kz");

  klee_make_symbolic(x, numX * sizeof(float), "x");
  klee_make_symbolic(y, numX * sizeof(float), "y");
  klee_make_symbolic(z, numX * sizeof(float), "z");

  klee_make_symbolic(phiR, numK * sizeof(float), "phiR");
  klee_make_symbolic(phiI, numK * sizeof(float), "phiI");

  klee_make_symbolic(dR, numK * sizeof(float), "dR");
  klee_make_symbolic(dI, numK * sizeof(float), "dI");

  /* Create CPU data structures */
  createDataStructs(numK, numX, realRhoPhi, imagRhoPhi, cpuOutR, cpuOutI);
  ComputeRhoPhi(numK, phiR, phiI, dR, dI, realRhoPhi, imagRhoPhi);
  kVals = (kValues*)calloc(numK, sizeof (kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].RealRhoPhi = realRhoPhi[k];
    kVals[k].ImagRhoPhi = imagRhoPhi[k];
  }

  /* Main computation */
  ComputeFH(numK, numX, kVals, x, y, z, cpuOutR, cpuOutI);

  computeMRIFHD_GPU(512, 256, numK, numX, kx, ky, kz, x, y, z, phiR, phiI,
                    dR, dI, &gpuOutR, &gpuOutI);

  int same = 1;
  for (int x = 0; x < numX; ++x) {
    same &= float_bitwise_eq(cpuOutR[x], gpuOutR[x]);
    same &= float_bitwise_eq(cpuOutI[x], gpuOutI[x]);
  }
  klee_print_expr("same", same);

  assert(same);
}

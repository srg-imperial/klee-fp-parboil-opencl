#include <assert.h>
#include <stdlib.h>
#include <klee/klee.h>
#include <klee-bits.h>

#include "../base/structs.h"
#include "../base/cenergy.h"
#include "../opencl/cuenergy.h"

int main(int argc, char **argv) {
  unsigned numatoms = 2;
  voldim3i grid = { 16, 16, 1 };
  size_t volsize[3] = { grid.x, grid.y, grid.z };
  size_t localWorkSize[3] = { grid.x/2, grid.y/2, 1 };

  float gridspacing;
  float *atoms = (float *) malloc(4*numatoms * sizeof(float));
  float *cenergy = (float *) malloc(grid.x * grid.y * grid.z * sizeof(float)), *genergy;

  klee_make_symbolic(&gridspacing, sizeof(gridspacing), "gridspacing");
  klee_make_symbolic(atoms, 4*numatoms * sizeof(float), "atoms");

  cpuenergy(grid, numatoms, gridspacing, 0, atoms, cenergy);
  gpuenergy(volsize, volsize, localWorkSize, numatoms, gridspacing, atoms, genergy);
  
  int same;
  for (int x = 0; x < grid.x * grid.y * grid.z; ++x) {
    same &= float_bitwise_eq(cenergy[x], genergy[x]);
  }
  klee_print_expr("same", same);

  assert(same);
}


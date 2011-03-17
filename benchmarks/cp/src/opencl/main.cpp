/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
/*
 * CUDA accelerated coulombic potential grid test code
 *   John E. Stone <johns@ks.uiuc.edu>
 *   http://www.ks.uiuc.edu/~johns/
 *
 * Coulombic potential grid calculation microbenchmark based on the time
 * consuming portions of the 'cionize' ion placement tool.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<algorithm>

#include "common.h"

#include "cuenergy.h"

/* initatoms()
 * Store a pseudorandom arrangement of point charges in *atombuf.
 */
static int
initatoms(float **atombuf, int count, size_t* volsize, float gridspacing) {
  size_t size[3];
  int i;
  float *atoms;

  srand(54321);			// Ensure that atom placement is repeatable

  atoms = (float *) malloc(count * 4 * sizeof(float));
  *atombuf = atoms;

  // compute grid dimensions in angstroms
  size[0] = gridspacing * volsize[0];
  size[1] = gridspacing * volsize[1];
  size[2] = gridspacing * volsize[2];

  for (i=0; i<count; i++) {
    int addr = i * 4;
    atoms[addr    ] = (rand() / (float) RAND_MAX) * size[0]; 
    atoms[addr + 1] = (rand() / (float) RAND_MAX) * size[1]; 
    atoms[addr + 2] = (rand() / (float) RAND_MAX) * size[2]; 
    atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
  }  

  return 0;
}

/* writeenergy()
 * Write part of the energy array to an output file for verification.
 */
static int
writeenergy(char *filename, float *energy, size_t* volsize)
{
  FILE *outfile;
  int x, y;

  outfile = fopen(filename, "w");
  if (outfile == NULL) {
    fputs("Cannot open output file\n", stderr);
    return -1;
    }

  /* Print the execution parameters */
  fprintf(outfile, "%d %d %d %d\n", volsize[0], volsize[1], volsize[2], ATOMCOUNT);

  /* Print a checksum */
  {
    double sum = 0.0;

    for (y = 0; y < volsize[1]; y++) {
      for (x = 0; x < volsize[0]; x++) {
        double t = energy[y*volsize[0]+x];
        t = std::max(-20.0, std::min(20.0, t));
    	sum += t;
      }
    }
    fprintf(outfile, "%.4g\n", sum);
  }
  
  /* Print several rows of the computed data */
  for (y = 0; y < 17; y++) {
    for (x = 0; x < volsize[0]; x++) {
      int addr = y * volsize[0] + x;
      fprintf(outfile, "%.4g ", energy[addr]);
    }
    fprintf(outfile, "\n");
  }

  fclose(outfile);

  return 0;
}


int main(int argc, char** argv) {

  float *energy = NULL;		// Output of device calculation
  float *atoms = NULL;
  size_t volsize[3], globalWorkSize[3], localWorkSize[3];

  // number of atoms to simulate (default)
  int atomcount = ATOMCOUNT;
  // setup energy grid size (default)
  volsize[0] = VOLSIZEX;
  volsize[1] = VOLSIZEY;
  volsize[2] = 1;

  // use user-specified values if provided
  getCmdLineParamInt("-volx", argc, argv, (int*)&volsize[0]);
  getCmdLineParamInt("-voly", argc, argv, (int*)&volsize[1]);
  getCmdLineParamInt("-atoms", argc, argv, &atomcount);

  // voxel spacing
  const float gridspacing = 0.1;

  //printf("CUDA accelerated coulombic potential microbenchmark\n");
  //printf("Original version by John E. Stone <johns@ks.uiuc.edu>\n");
  //printf("This version maintained by Chris Rodrigues\n");

  // setup CUDA grid and block sizes (default)
  localWorkSize[0] = BLOCKSIZEX;		// each thread does multiple Xs
  localWorkSize[1] = BLOCKSIZEY;
  localWorkSize[2] = 1;
  globalWorkSize[0] = volsize[0] / UNROLLX; // each thread does multiple Xs
  globalWorkSize[1] = volsize[1]; 
  globalWorkSize[2] = volsize[2]; 

  // use user-specified block sizes if provided
  getCmdLineParamInt("-localx", argc, argv, (int*)&localWorkSize[0]);
  getCmdLineParamInt("-localy", argc, argv, (int*)&localWorkSize[1]);

  // allocate and initialize atom coordinates and charges
  if (initatoms(&atoms, atomcount, volsize, gridspacing))
    return -1;

  if (gpuenergy(volsize, globalWorkSize, localWorkSize, atomcount, gridspacing, atoms, energy))
    return -1;

#ifdef PROFILING
  // output kernel runtimes
  double avg;
  for (int i=0; i<numIterations; i++)
  {
    cl_ulong culStart, culEnd;
    clGetEventProfilingInfo(evKernel[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
    clGetEventProfilingInfo(evKernel[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);

    double diff = (double)(culEnd-culStart)/1000000;
    avg += diff;

    printf ("kernel %d: %fms\n", i, diff);
  }
  printf ("kernel avg: %fms\n", avg/numIterations);
#endif

  free(atoms);
  free(energy);

  return 0;
}

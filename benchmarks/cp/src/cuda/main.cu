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
#include <sys/time.h>

#include <common.h>

#include "cuenergy.h"

/* initatoms()
 * Store a pseudorandom arrangement of point charges in *atombuf.
 */
static int
initatoms(float **atombuf, int count, dim3 volsize, float gridspacing) {
  dim3 size;
  int i;
  float *atoms;

  srand(54321);			// Ensure that atom placement is repeatable

  atoms = (float *) malloc(count * 4 * sizeof(float));
  *atombuf = atoms;

  // compute grid dimensions in angstroms
  size.x = gridspacing * volsize.x;
  size.y = gridspacing * volsize.y;
  size.z = gridspacing * volsize.z;

  for (i=0; i<count; i++) {
    int addr = i * 4;
    atoms[addr    ] = (rand() / (float) RAND_MAX) * size.x; 
    atoms[addr + 1] = (rand() / (float) RAND_MAX) * size.y; 
    atoms[addr + 2] = (rand() / (float) RAND_MAX) * size.z; 
    atoms[addr + 3] = ((rand() / (float) RAND_MAX) * 2.0) - 1.0;  // charge
  }  

  return 0;
}

/* writeenergy()
 * Write part of the energy array to an output file for verification.
 */
static int
writeenergy(char *filename, float *energy, dim3 volsize)
{
  FILE *outfile;
  int x, y;

  outfile = fopen(filename, "w");
  if (outfile == NULL) {
    fputs("Cannot open output file\n", stderr);
    return -1;
    }

  /* Print the execution parameters */
  fprintf(outfile, "%d %d %d %d\n", volsize.x, volsize.y, volsize.z, ATOMCOUNT);

  /* Print a checksum */
  {
    double sum = 0.0;

    for (y = 0; y < volsize.y; y++) {
      for (x = 0; x < volsize.x; x++) {
        double t = energy[y*volsize.x+x];
        t = fmax(-20.0, fmin(20.0, t));
    	sum += t;
      }
    }
    fprintf(outfile, "%.4g\n", sum);
  }
  
  /* Print several rows of the computed data */
  for (y = 0; y < 17; y++) {
    for (x = 0; x < volsize.x; x++) {
      int addr = y * volsize.x + x;
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
  dim3 volsize, Gsz, Bsz;

  // number of atoms to simulate
  int atomcount = ATOMCOUNT;

  // voxel spacing
  const float gridspacing = 0.1;

  // Size of buffer on GPU
  int volmemsz;

  //printf("CUDA accelerated coulombic potential microbenchmark\n");
  //printf("Original version by John E. Stone <johns@ks.uiuc.edu>\n");
  //printf("This version maintained by Chris Rodrigues\n");

  // setup energy grid size
  volsize.x = VOLSIZEX;
  volsize.y = VOLSIZEY;
  volsize.z = 1;

  // use user-specified values if provided
  getCmdLineParamInt("-volx", argc, argv, (int*)&volsize.x);
  getCmdLineParamInt("-voly", argc, argv, (int*)&volsize.y);
  getCmdLineParamInt("-atoms", argc, argv, &atomcount);

  // setup CUDA grid and block sizes
  Bsz.x = BLOCKSIZEX;		// each thread does multiple Xs
  Bsz.y = BLOCKSIZEY;
  Bsz.z = 1;
  // use user-specified block sizes if provided
  getCmdLineParamInt("-localx", argc, argv, (int*)&Bsz.x);
  getCmdLineParamInt("-localy", argc, argv, (int*)&Bsz.y);

  Gsz.x = volsize.x / (Bsz.x * UNROLLX); // each thread does multiple Xs
  Gsz.y = volsize.y / Bsz.y; 
  Gsz.z = volsize.z / Bsz.z; 

#if 0
  printf("Grid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);
  printf("Running kernel(atoms:%d, gridspacing %g, z %d)\n", atomcount, gridspacing, 0);
#endif

  // allocate and initialize atom coordinates and charges
  if (initatoms(&atoms, atomcount, volsize, gridspacing))
    return -1;

  // allocate and initialize the GPU output array
  volmemsz = sizeof(float) * volsize.x * volsize.y * volsize.z;

#ifdef PROFILING
  // profile kernel executions
  int numIterations = (atomcount%MAXATOMS) ? (atomcount/MAXATOMS + 1) : (atomcount/MAXATOMS);
  cudaEvent_t *evStart = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * numIterations);
  cudaEvent_t *evStop  = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * numIterations);
  for (int i=0; i<numIterations; i++)
  {
    cudaEventCreate(&evStart[i]);
    cudaEventCreate(&evStop[i]);
  }

  inf_timer tMain;
  startTimer(&tMain);
#endif

  // Main computation
  {
    float *d_output = NULL;	// Output on device
    int iterations=0;
    int atomstart;

    cudaMalloc((void**)&d_output, volmemsz);
    cudaMemset(d_output, 0, volmemsz);

    for (atomstart=0; atomstart<atomcount; atomstart+=MAXATOMS) {   
      int atomsremaining = atomcount - atomstart;
      int runatoms = (atomsremaining > MAXATOMS) ? MAXATOMS : atomsremaining;
      iterations++;

      // copy the atoms to the GPU
      if (copyatomstoconstbuf(atoms + 4*atomstart, runatoms, 0*gridspacing)) 
	return -1;

      //if (parameters->synchronizeGpu) cudaThreadSynchronize();

#ifdef PROFILING
      cudaEventRecord(evStart[iterations-1], 0);
#endif
 
      // RUN the kernel...
      cenergy<<<Gsz, Bsz, 0>>>(runatoms, 0.1, d_output);

#ifdef PROFILING
      cudaEventRecord(evStop[iterations-1], 0);
#endif

    }

    // Copy the GPU output data back to the host and use/store it..
    energy = (float *) malloc(volmemsz);
    cudaMemcpy(energy, d_output, volmemsz,  cudaMemcpyDeviceToHost);

    cudaFree(d_output);
  }

#ifdef PROFILING
  cudaThreadSynchronize();
  stopTimer(&tMain);
  printf ("main computation: %10fms\n", elapsedTime(tMain));
#endif

  /* Print a subset of the results to a file */
  /*if (parameters->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    if (writeenergy(parameters->outFile, energy, volsize) == -1)
      return -1;
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }*/

  free(atoms);
  free(energy);

#ifdef PROFILING
  // output kernel runtimes
  float avg;
  for (int i=0; i<numIterations; i++)
  {
    float diff;
    cudaEventElapsedTime(&diff, evStart[i], evStop[i]);
    printf ("kernel %d: %fms\n", i, diff);

    avg += diff;
  }
  printf ("kernel avg: %fms\n", avg/numIterations);

  free(evStart);
  free(evStop);
#endif

  return 0;
}




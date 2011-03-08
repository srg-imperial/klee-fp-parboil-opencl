/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/*
 * CUDA code for creating the FHD data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 * 
 * recommended g++ options:
 *   -O3 -lm -ffast-math -funroll-all-loops
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *     function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 */

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

#include <common.h>
#include <ocl-wrapper.h>

#include "file.h"

#include "computeFH.h"


#define KERNEL_RHO_PHI_THREADS_PER_BLOCK 512
#define KERNEL_FH_THREADS_PER_BLOCK 256
#define KERNEL_FH_K_ELEMS_PER_GRID 512

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

int
main (int argc, char *argv[])
{
  int numX, numK;		/* Number of X and K values */
  int original_numK, original_numX;		/* Number of K and X values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *dR, *dI;               /* D values (complex) */
  float *outI, *outR;		/* Output signal (complex) */
  size_t rhophiWorkSize, fhWorkSize;

  /* Read in data */
  //inputData(params->inpFiles[0],
  inputData(argv[1],
	    &original_numK, &original_numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI,
	    &dR, &dI);

  /* Reduce the number of k-space samples if a number is given
   * on the command line */
  if (getCmdLineParamInt("-numK", argc, argv, &numK))
    numK = MIN(numK, original_numK);
  else
    numK = original_numK;

  if (getCmdLineParamInt("-numX", argc, argv, &numX))
    numX = MIN(numX, original_numX);
  else
    numX = original_numX;

  rhophiWorkSize = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
  getCmdLineParamInt("-rhophi-local", argc, argv, (int*)&rhophiWorkSize);

  fhWorkSize = KERNEL_FH_THREADS_PER_BLOCK;
  getCmdLineParamInt("-fh-local", argc, argv, (int*)&fhWorkSize);

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  if (computeMRIFHD_GPU(rhophiWorkSize, fhWorkSize, numK, numX, kx, ky, kz,
                        x, y, z, phiR, phiI, dR, dI, &outR, &outI))
    return 1;

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (dR);
  free (dI);
  free (outR);
  free (outI);

  return 0;
}

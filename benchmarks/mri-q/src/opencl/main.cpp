/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

/* 
 * C code for creating the Q data structure for fast convolution-based 
 * Hessian multiplication for arbitrary k-space trajectories.
 *
 * Inputs:
 * kx - VECTOR of kx values, same length as ky and kz
 * ky - VECTOR of ky values, same length as kx and kz
 * kz - VECTOR of kz values, same length as kx and ky
 * x  - VECTOR of x values, same length as y and z
 * y  - VECTOR of y values, same length as x and z
 * z  - VECTOR of z values, same length as x and y
 * phi - VECTOR of the Fourier transform of the spatial basis 
 *      function, evaluated at [kx, ky, kz].  Same length as kx, ky, and kz.
 *
 * recommended g++ options:
 *  -O3 -lm -ffast-math -funroll-all-loops
 */

#include <stddef.h>
#include <stdlib.h>

#include <common.h>
#include "file.h"

#include "computeMRIQ.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 512
#define KERNEL_Q_THREADS_PER_BLOCK 256

int
main (int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK, original_numX;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *Qr, *Qi;		/* Q signal (complex) */
  size_t phiWorkSize, qWorkSize;


  /* Read in data */
  inputData(argv[1],
	    &original_numK, &original_numX,
	    &kx, &ky, &kz,
	    &x, &y, &z,
	    &phiR, &phiI);

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

  phiWorkSize = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  getCmdLineParamInt("-phimag-local", argc, argv, (int*)&phiWorkSize);

  qWorkSize = KERNEL_Q_THREADS_PER_BLOCK;
  getCmdLineParamInt("-q-local", argc, argv, (int*)&qWorkSize);

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  if (computeMRIQ_GPU(phiWorkSize, qWorkSize, numK, numX, kx, ky, kz, x, y, z, phiR, phiI, &Qr, &Qi))
    return 1;

  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (Qr);
  free (Qi);

  return 0;
}

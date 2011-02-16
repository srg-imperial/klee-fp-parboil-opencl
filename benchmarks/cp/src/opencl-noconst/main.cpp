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
#include<algorithm>

#include <CL/cl.h>

#include "common.h"
#include "ocl-wrapper.h"

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
// This function copies atoms from the CPU to the GPU and
// precalculates (z^2) for each atom.

int copyatomstoconstbuf(cl_command_queue cmdQueue, cl_mem d_atoms, float *atoms, int count, float zplane) {

  float atompre[4*count];
  int i;
  for (i=0; i<count*4; i+=4) {
    atompre[i    ] = atoms[i    ];
    atompre[i + 1] = atoms[i + 1];
    float dz = zplane - atoms[i + 2];
    atompre[i + 2]  = dz*dz;
    atompre[i + 3] = atoms[i + 3];
  }

  clEnqueueWriteBuffer(cmdQueue, d_atoms, CL_TRUE, 0, count * 4 * sizeof(float), atompre, 0, NULL, NULL);

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

  // Size of buffer on GPU
  int volmemsz;

  //printf("CUDA accelerated coulombic potential microbenchmark\n");
  //printf("Original version by John E. Stone <johns@ks.uiuc.edu>\n");
  //printf("This version maintained by Chris Rodrigues\n");

  // initialize OpenCL context and create command queue
  cl_int ciErr;
  OclWrapper *ocl = new OclWrapper;
  if (!ocl->init (OPENCL_PLATFORM, OPENCL_DEVICE_ID))
    return 1;

  // load and build kernel
  cl_program program;
  if (!ocl->buildProgram ("cuenergy_pre8_coalesce-noconst.cl", &program))
    return 1;
  cl_kernel kernel = clCreateKernel(program, "cenergy", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create kernel")) return 1;

  // setup CUDA grid and block sizes (default)
  localWorkSize[0] = BLOCKSIZEX;		// each thread does multiple Xs
  localWorkSize[1] = BLOCKSIZEY;
  localWorkSize[2] = 1;
  globalWorkSize[0] = volsize[0] / UNROLLX; // each thread does multiple Xs
  globalWorkSize[1] = volsize[1]; 
  globalWorkSize[3] = volsize[2]; 

  // use user-specified block sizes if provided
  getCmdLineParamInt("-localx", argc, argv, (int*)&localWorkSize[0]);
  getCmdLineParamInt("-localy", argc, argv, (int*)&localWorkSize[1]);

#if 0
  printf("Grid size: %d x %d x %d\n", volsize.x, volsize.y, volsize.z);
  printf("Running kernel(atoms:%d, gridspacing %g, z %d)\n", atomcount, gridspacing, 0);
#endif

  // allocate and initialize atom coordinates and charges
  if (initatoms(&atoms, atomcount, volsize, gridspacing))
    return -1;

  // allocate and initialize the GPU output array
  volmemsz = sizeof(float) * volsize[0] * volsize[1] * volsize[2];

  // profile kernel executions
  cl_event evKernel;

  inf_timer tMain;
  startTimer(&tMain);

  // Main computation
  {
    cl_mem d_output;	// Output on device
    cl_mem d_atoms;     // memory to store constant atom data
    int iterations=0;
    int atomstart;
    cl_int ciErr;

    // create output array and initialize it with 0s
    d_output = clCreateBuffer(ocl->getContext(), CL_MEM_WRITE_ONLY, volmemsz, NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create output buffer");

    energy = (float*)malloc(volmemsz);
    memset(energy, 0, volmemsz);
    clEnqueueWriteBuffer(ocl->getCmdQueue(), d_output, CL_TRUE, 0, volmemsz, energy, 0, NULL, NULL);

    // create memory buffer for atoms
    d_atoms = clCreateBuffer(ocl->getContext(), CL_MEM_READ_ONLY, atomcount*4*sizeof(float), NULL, &ciErr);


    // set kernel arguments
    float gridspacing = 0.1;
    ciErr = clSetKernelArg(kernel, 1, sizeof(float), (void*)&gridspacing);
    ciErr = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_output);
    ciErr = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_atoms);

    //for (atomstart=0; atomstart<atomcount; atomstart+=MAXATOMS) {   
    //  int atomsremaining = atomcount - atomstart;
    //  int runatoms = (atomsremaining > MAXATOMS) ? MAXATOMS : atomsremaining;
    //  iterations++;

      // copy the atoms to the GPU
      if (copyatomstoconstbuf(ocl->getCmdQueue(), d_atoms, atoms, atomcount, 0*gridspacing)) 
        return -1;

      //if (parameters->synchronizeGpu) cudaThreadSynchronize();

      // set dynamic kernel arguments
      ciErr = clSetKernelArg(kernel, 0, sizeof(int), (void*)&atomcount);
 
      // RUN the kernel...
      ciErr = clEnqueueNDRangeKernel (ocl->getCmdQueue(), kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &evKernel);

    //}

    // Copy the GPU output data back to the host
    ciErr = clEnqueueReadBuffer(ocl->getCmdQueue(), d_output, CL_TRUE, 0, volmemsz, energy, 0, NULL, NULL);

    clReleaseMemObject(d_output);
    clReleaseMemObject(d_atoms);
  }

  stopTimer(&tMain);
  printf ("main computation: %fms\n", elapsedTime(tMain));

  /* Print a subset of the results to a file */
  /*if (parameters->outFile) {
    pb_SwitchToTimer(&timers, pb_TimerID_IO);
    if (writeenergy(parameters->outFile, energy, volsize) == -1)
      return -1;
    pb_SwitchToTimer(&timers, pb_TimerID_COMPUTE);
  }*/


  free(atoms);
  free(energy);

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  delete ocl;

  // output kernel runtimes
  cl_ulong culStart, culEnd;
  clGetEventProfilingInfo(evKernel, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
  clGetEventProfilingInfo(evKernel, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);

  double diff = (double)(culEnd-culStart)/1000000;

  printf ("kernel: %fms\n", diff);

  return 0;
}




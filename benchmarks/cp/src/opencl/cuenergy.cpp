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

#include <CL/cl.h>

#include "common.h"
#include "ocl-wrapper.h"

#include "cuenergy.h"

// This function copies atoms from the CPU to the GPU and
// precalculates (z^2) for each atom.

int copyatomstoconstbuf(cl_command_queue cmdQueue, cl_mem d_atoms, const float *atoms, int count, float zplane) {
  if (count > MAXATOMS) {
    printf("Atom count exceeds constant buffer storage capacity\n");
    return -1;
  }

  float atompre[4*MAXATOMS];
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


int gpuenergy(size_t volsize[3], size_t globalWorkSize[3], size_t localWorkSize[3],
              int atomcount, float gridspacing, const float *atoms, float *&energy) {
  // Size of buffer on GPU
  int volmemsz;

  // initialize OpenCL context and create command queue
  cl_int ciErr;
  OclWrapper *ocl = new OclWrapper;
  if (!ocl->init (OPENCL_PLATFORM, OPENCL_DEVICE_ID))
    return 1;

  // load and build kernel
  cl_program program;
  if (!ocl->buildProgram ("cuenergy_pre8_coalesce.cl", &program))
    return 1;
  cl_kernel kernel = clCreateKernel(program, "cenergy", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create kernel")) return 1;

  // allocate and initialize the GPU output array
  volmemsz = sizeof(float) * volsize[0] * volsize[1] * volsize[2];

  // profile kernel executions
  int numIterations = (atomcount%MAXATOMS) ? (atomcount/MAXATOMS + 1) : (atomcount/MAXATOMS);
  cl_event *evKernel = (cl_event*) malloc (sizeof(cl_event) * numIterations);

#ifdef PROFILING
  inf_timer tMain;
  startTimer(&tMain);
#endif

  // Main computation
  {
    cl_mem d_output;	// Output on device
    cl_mem d_atoms;     // memory to store constant atom data
    int iterations=0;
    int atomstart;
    cl_int ciErr;

    // create output array and initialize it with 0s
    d_output = clCreateBuffer(ocl->getContext(), CL_MEM_READ_WRITE, volmemsz, NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create output buffer");

    energy = (float*)malloc(volmemsz);
    memset(energy, 0, volmemsz);
    clEnqueueWriteBuffer(ocl->getCmdQueue(), d_output, CL_TRUE, 0, volmemsz, energy, 0, NULL, NULL);

    // create constant memory buffer for atoms
    d_atoms = clCreateBuffer(ocl->getContext(), CL_MEM_READ_ONLY, MAXATOMS*4*sizeof(float), NULL, &ciErr);


    // set kernel arguments
    float gridspacing = 0.1;
    ciErr = clSetKernelArg(kernel, 1, sizeof(float), (void*)&gridspacing);
    ciErr = clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_output);
    ciErr = clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_atoms);

    for (atomstart=0; atomstart<atomcount; atomstart+=MAXATOMS) {   
      int atomsremaining = atomcount - atomstart;
      int runatoms = (atomsremaining > MAXATOMS) ? MAXATOMS : atomsremaining;
      iterations++;

      // copy the atoms to the GPU
      if (copyatomstoconstbuf(ocl->getCmdQueue(), d_atoms, atoms + 4*atomstart, runatoms, 0*gridspacing)) 
        return -1;

      //if (parameters->synchronizeGpu) cudaThreadSynchronize();

      // set dynamic kernel arguments
      ciErr = clSetKernelArg(kernel, 0, sizeof(int), (void*)&runatoms);
 
      // RUN the kernel...
      ciErr = clEnqueueNDRangeKernel (ocl->getCmdQueue(), kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, &evKernel[iterations-1]);

    }
#if 0
    printf("Done\n");
#endif

    // Copy the GPU output data back to the host
    ciErr = clEnqueueReadBuffer(ocl->getCmdQueue(), d_output, CL_TRUE, 0, volmemsz, energy, 0, NULL, NULL);

    clReleaseMemObject(d_output);
    clReleaseMemObject(d_atoms);
  }

#ifdef PROFILING
  stopTimer(&tMain);
  printf ("main computation: %fms\n", elapsedTime(tMain));
#endif

  clReleaseKernel(kernel);
  clReleaseProgram(program);
  delete ocl;

  free(evKernel);

  return 0;
}

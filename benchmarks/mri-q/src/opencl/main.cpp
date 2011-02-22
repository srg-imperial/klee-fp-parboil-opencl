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

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <malloc.h>
#include <string.h>
#include <algorithm>

#include <common.h>
#include <ocl-wrapper.h>

#include "file.h"

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define K_ELEMS_PER_GRID 2048

#define KERNEL_PHI_MAG_THREADS_PER_BLOCK 512
#define KERNEL_Q_THREADS_PER_BLOCK 256
#define KERNEL_Q_K_ELEMS_PER_GRID 1024

struct kValues {
  float Kx; 
  float Ky; 
  float Kz; 
  float PhiMag;
};

cl_event evPhiMag, *evQ;
int numQ;


static void
setupMemoryGPU(cl_context ctxt, cl_command_queue cq, int num, int size, cl_mem& dev_m, float*& host_ptr)
{
  cl_int ciErr;
  dev_m = clCreateBuffer(ctxt, CL_MEM_READ_ONLY, num*size, NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer");
  ciErr = clEnqueueWriteBuffer(cq, dev_m, CL_TRUE, 0, num*size, host_ptr, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "write buffer");
}

static void
cleanupMemoryGPU(cl_command_queue cq, int num, int size, cl_mem & dev_m, float * host_ptr)
{
  cl_int ciErr;
  ciErr = clEnqueueReadBuffer(cq, dev_m, CL_TRUE, 0, num*size, host_ptr, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "read buffer");
  ciErr = clReleaseMemObject(dev_m);
  OclWrapper::checkErr(ciErr, "release buffer");
}

void computePhiMag_GPU(cl_command_queue cq, cl_kernel ckPhiMag, int argc, char** argv, int numK, cl_mem * phiR_d, cl_mem * phiI_d, cl_mem * phiMag_d)
{
  size_t localWorkSize;
  size_t globalWorkSize;

  localWorkSize = KERNEL_PHI_MAG_THREADS_PER_BLOCK;
  getCmdLineParamInt("-phimag-local", argc, argv, (int*)&localWorkSize);

  int phiMagBlocks = numK / localWorkSize;
  if (numK % localWorkSize)
    phiMagBlocks++;

  globalWorkSize = phiMagBlocks * localWorkSize;

  cl_int ciErr;
  ciErr  = clSetKernelArg (ckPhiMag, 0, sizeof(cl_mem), phiR_d);
  ciErr |= clSetKernelArg (ckPhiMag, 1, sizeof(cl_mem), phiI_d);
  ciErr |= clSetKernelArg (ckPhiMag, 2, sizeof(cl_mem), phiMag_d);
  ciErr |= clSetKernelArg (ckPhiMag, 3, sizeof(int), (void*)&numK);
  OclWrapper::checkErr(ciErr, "computePhiMag: set kernel args");

  ciErr = clEnqueueNDRangeKernel(cq, ckPhiMag, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &evPhiMag);
  OclWrapper::checkErr(ciErr, "computePhiMag: start kernel");
}

void computeQ_GPU(cl_command_queue cq, cl_kernel ckQ, int argc, char** argv,
                  int numK, int numX,
                  cl_mem * x_d, cl_mem * y_d, cl_mem * z_d,
                  kValues* kVals,
                  cl_mem * Qr_d, cl_mem * Qi_d,
                  cl_mem * c_d)
{
  size_t localWorkSize;
  size_t globalWorkSize;

  localWorkSize = KERNEL_Q_THREADS_PER_BLOCK;
  getCmdLineParamInt("-q-local", argc, argv, (int*)&localWorkSize);

  int QGrids = numK / KERNEL_Q_K_ELEMS_PER_GRID;
  if (numK % KERNEL_Q_K_ELEMS_PER_GRID)
    QGrids++;
  int QBlocks = numX / localWorkSize;
  if (numX % localWorkSize)
    QBlocks++;

  globalWorkSize = QBlocks * localWorkSize;

  numQ = QGrids;
  evQ = (cl_event*) malloc (sizeof(cl_event) * numQ);

  cl_int ciErr;
  ciErr  = clSetKernelArg (ckQ, 0, sizeof(int), (void*)&numK);
  ciErr |= clSetKernelArg (ckQ, 2, sizeof(cl_mem), x_d);
  ciErr |= clSetKernelArg (ckQ, 3, sizeof(cl_mem), y_d);
  ciErr |= clSetKernelArg (ckQ, 4, sizeof(cl_mem), z_d);
  ciErr |= clSetKernelArg (ckQ, 5, sizeof(cl_mem), Qr_d);
  ciErr |= clSetKernelArg (ckQ, 6, sizeof(cl_mem), Qi_d);
  ciErr |= clSetKernelArg (ckQ, 7, sizeof(cl_mem), c_d);
  OclWrapper::checkErr(ciErr, "computeQ: set kernel args");

  // KERNEL_Q_K_ELEMS_PER_GRID);
  for (int QGrid = 0; QGrid < QGrids; QGrid++) {
    // Put the tile of K values into constant mem
    int QGridBase = QGrid * KERNEL_Q_K_ELEMS_PER_GRID;
    kValues* kValsTile = kVals + QGridBase;
    int numElems = MIN(KERNEL_Q_K_ELEMS_PER_GRID, numK - QGridBase);

    clSetKernelArg (ckQ, 1, sizeof(int), (void*)&QGridBase);

    ciErr = clEnqueueWriteBuffer(cq, *c_d, CL_TRUE, 0, numElems * sizeof(kValues), kValsTile, 0, NULL, NULL);
    OclWrapper::checkErr (ciErr, "write constant buffer c");

    ciErr = clEnqueueNDRangeKernel(cq, ckQ, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &evQ[QGrid]);
    OclWrapper::checkErr(ciErr, "computeQ: start kernel");
  }
}

void createDataStructsCPU(int numK, int numX, float** phiMag,
     float** Qr, float** Qi)
{
  *phiMag = (float* ) memalign(16, numK * sizeof(float));
  *Qr = (float*) memalign(16, numX * sizeof (float));
  *Qi = (float*) memalign(16, numX * sizeof (float));
}


int
main (int argc, char *argv[]) {
  int numX, numK;		/* Number of X and K values */
  int original_numK, original_numX;		/* Number of K values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *phiMag;		/* Magnitude of Phi */
  float *Qr, *Qi;		/* Q signal (complex) */

  struct kValues* kVals;


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

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  // initialize OpenCL context and create command queue
  cl_int ciErr;
  OclWrapper *ocl = new OclWrapper;
  if (!ocl->init (OPENCL_PLATFORM, OPENCL_DEVICE_ID))
    return 1;

  // load and build kernels
  cl_program program;
  if (!ocl->buildProgram ("computeQ.cl", &program))
    return 1;
  cl_kernel ckComputePhiMag = clCreateKernel(program, "ComputePhiMag", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create kernel PhiMag")) return 1;
  cl_kernel ckComputeQ = clCreateKernel(program, "ComputeQ", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create kernel Q")) return 1;


  /* Create CPU data structures */
  createDataStructsCPU(numK, numX, &phiMag, &Qr, &Qi);

  /* GPU section 1 (precompute PhiMag) */
  {
    /* Mirror several data structures on the device */
    cl_mem phiR_d, phiI_d;
    cl_mem phiMag_d;

    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numK, sizeof(float), phiR_d, phiR);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numK, sizeof(float), phiI_d, phiI);
    phiMag_d = clCreateBuffer(ocl->getContext(), CL_MEM_WRITE_ONLY, numK*sizeof(float), NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create buffer: phiMag_d");

    //if (params->synchronizeGpu) cudaThreadSynchronize();

    computePhiMag_GPU(ocl->getCmdQueue(), ckComputePhiMag, argc, argv, numK, &phiR_d, &phiI_d, &phiMag_d);

    //if (params->synchronizeGpu) cudaThreadSynchronize();

    cleanupMemoryGPU(ocl->getCmdQueue(), numK, sizeof(float), phiMag_d, phiMag);
    clReleaseMemObject(phiR_d);
    clReleaseMemObject(phiI_d);
  }

  kVals = (struct kValues*)calloc(numK, sizeof (struct kValues));
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].PhiMag = phiMag[k];
  }

  free(phiMag);

  /* GPU section 2 */
  {
    cl_mem x_d, y_d, z_d;
    cl_mem Qr_d, Qi_d;
    cl_mem c_d;

    // copy input data to device
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numX, sizeof(float), x_d, x);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numX, sizeof(float), y_d, y);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numX, sizeof(float), z_d, z);

    // constant buffer
    c_d = clCreateBuffer(ocl->getContext(), CL_MEM_READ_ONLY, sizeof(kValues) * KERNEL_Q_K_ELEMS_PER_GRID, NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create constant buffer");

    // initialize output with 0
    Qr_d = clCreateBuffer(ocl->getContext(), CL_MEM_READ_WRITE, numX*sizeof(float), NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create buffer: Qr_d");
    Qi_d = clCreateBuffer(ocl->getContext(), CL_MEM_READ_WRITE, numX*sizeof(float), NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create buffer: Qi_d");
    memset(Qr, 0, numX*sizeof(float));
    memset(Qi, 0, numX*sizeof(float));
    ciErr  = clEnqueueWriteBuffer(ocl->getCmdQueue(), Qr_d, CL_TRUE, 0, numX * sizeof(float), Qr, 0, NULL, NULL);
    ciErr |= clEnqueueWriteBuffer(ocl->getCmdQueue(), Qi_d, CL_TRUE, 0, numX * sizeof(float), Qi, 0, NULL, NULL);
    OclWrapper::checkErr(ciErr, "write buffer Qr, Qi");


    //if (params->synchronizeGpu) cudaThreadSynchronize();

    computeQ_GPU(ocl->getCmdQueue(), ckComputeQ, argc, argv, numK, numX, &x_d, &y_d, &z_d, kVals, &Qr_d, &Qi_d, &c_d);

    //if (params->synchronizeGpu) cudaThreadSynchronize();

    clReleaseMemObject(x_d);
    clReleaseMemObject(y_d);
    clReleaseMemObject(z_d);
    cleanupMemoryGPU(ocl->getCmdQueue(), numX, sizeof(float), Qr_d, Qr);
    cleanupMemoryGPU(ocl->getCmdQueue(), numX, sizeof(float), Qi_d, Qi);
  }


  free (kx);
  free (ky);
  free (kz);
  free (x);
  free (y);
  free (z);
  free (phiR);
  free (phiI);
  free (kVals);
  free (Qr);
  free (Qi);

  clReleaseKernel(ckComputeQ);
  clReleaseKernel(ckComputePhiMag);
  clReleaseProgram(program);
  delete ocl;


#ifdef PROFILING
  printf ("\n-------- PROFILING RESULTS --------\n");
  // output kernel runtimes
  double avg, diff;
  cl_ulong culStart, culEnd;

  clGetEventProfilingInfo(evPhiMag, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
  clGetEventProfilingInfo(evPhiMag, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);
  diff = (double)(culEnd-culStart)/1000000;
  printf ("kernel computePhiMag: %fms\n", diff);

  for (int i=0; i<numQ; i++)
  {
    clGetEventProfilingInfo(evQ[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
    clGetEventProfilingInfo(evQ[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);

    diff = (double)(culEnd-culStart)/1000000;
    avg += diff;

    printf ("kernel computeQ (%d): %fms\n", i, diff);
  }
  printf ("kernel computeQ avg: %fms\n", avg/numQ);
  printf ("\n");
#endif

  free(evQ);


  return 0;
}

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
#include <algorithm>

#include <common.h>
#include <ocl-wrapper.h>

#include "file.h"


#define KERNEL_RHO_PHI_THREADS_PER_BLOCK 512
#define KERNEL_FH_THREADS_PER_BLOCK 256
#define KERNEL_FH_K_ELEMS_PER_GRID 512

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float RhoPhiR;
  float RhoPhiI;
};

cl_event evRhoPhi, *evFH;
int numFH;


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

void createDataStructs(int numK, int numX, 
                       float*& realRhoPhi, float*& imagRhoPhi, 
                       float*& outR, float*& outI)
{
  realRhoPhi = (float* ) calloc(numK, sizeof(float));
  imagRhoPhi = (float* ) calloc(numK, sizeof(float));
  outR = (float*) calloc (numX, sizeof (float));
  outI = (float*) calloc (numX, sizeof (float));
}

void computeRhoPhi_GPU(cl_command_queue cq, cl_kernel ckRhoPhi, int argc, char **argv,
                       int numK, 
                       cl_mem * phiR_d, cl_mem * phiI_d, cl_mem * dR_d, cl_mem * dI_d,
                       cl_mem * realRhoPhi_d, cl_mem * imagRhoPhi_d)
{
  size_t localWorkSize;
  size_t globalWorkSize;

  localWorkSize = KERNEL_RHO_PHI_THREADS_PER_BLOCK;
  getCmdLineParamInt("-rhophi-local", argc, argv, (int*)&localWorkSize);

  int rhoPhiBlocks = numK / localWorkSize;

  if (numK % localWorkSize)
    rhoPhiBlocks++;

  globalWorkSize = rhoPhiBlocks * localWorkSize;
  //printf("Launch RhoPhi Kernel on GPU: Blocks (%d, %d), Threads Per Block %d\n",
  //       rhoPhiBlocks, 1, KERNEL_RHO_PHI_THREADS_PER_BLOCK);

  clSetKernelArg(ckRhoPhi, 0, sizeof(int), (void*)&numK);
  clSetKernelArg(ckRhoPhi, 1, sizeof(cl_mem), phiR_d);
  clSetKernelArg(ckRhoPhi, 2, sizeof(cl_mem), phiI_d);
  clSetKernelArg(ckRhoPhi, 3, sizeof(cl_mem), dR_d);
  clSetKernelArg(ckRhoPhi, 4, sizeof(cl_mem), dI_d);
  clSetKernelArg(ckRhoPhi, 5, sizeof(cl_mem), realRhoPhi_d);
  clSetKernelArg(ckRhoPhi, 6, sizeof(cl_mem), imagRhoPhi_d);

  cl_int ciErr = clEnqueueNDRangeKernel (cq, ckRhoPhi, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &evRhoPhi);
  OclWrapper::checkErr(ciErr, "launch RhoPhi kernel");
}

void computeFH_GPU(cl_command_queue cq, cl_kernel ckFH, int argc, char **argv,
                   int numK, int numX, 
                   cl_mem * x_d, cl_mem * y_d, cl_mem * z_d,
                   kValues* kVals,
                   cl_mem * outR_d, cl_mem * outI_d,
                   cl_mem * c_d)
{
  size_t localWorkSize;
  size_t globalWorkSize;

  localWorkSize = KERNEL_FH_THREADS_PER_BLOCK;
  getCmdLineParamInt("-fh-local", argc, argv, (int*)&localWorkSize);

  int FHGrids = numK / KERNEL_FH_K_ELEMS_PER_GRID;
  if (numK % KERNEL_FH_K_ELEMS_PER_GRID)
    FHGrids++;
  int FHBlocks = numX / localWorkSize;
  if (numX % localWorkSize)
    FHBlocks++;
  globalWorkSize = FHBlocks * localWorkSize;

  //printf("Launch GPU Kernel: Grids %d, Blocks Per Grid (%d, %d), Threads Per Block (%d, %d), K Elems Per Thread %d\n",
  //       FHGrids, DimFHGrid.x, DimFHGrid.y, DimFHBlock.x, DimFHBlock.y, KERNEL_FH_K_ELEMS_PER_GRID);

  clSetKernelArg(ckFH, 0, sizeof(int), (void*)&numK);
  clSetKernelArg(ckFH, 2, sizeof(cl_mem), x_d);
  clSetKernelArg(ckFH, 3, sizeof(cl_mem), y_d);
  clSetKernelArg(ckFH, 4, sizeof(cl_mem), z_d);
  clSetKernelArg(ckFH, 5, sizeof(cl_mem), outR_d);
  clSetKernelArg(ckFH, 6, sizeof(cl_mem), outI_d);
  clSetKernelArg(ckFH, 7, sizeof(cl_mem), c_d);

  numFH = FHGrids;
  evFH = (cl_event*) malloc (sizeof(cl_event) * numFH);

  cl_int ciErr;
  for (int FHGrid = 0; FHGrid < FHGrids; FHGrid++) {
    // Put the tile of K values into constant mem
    int FHGridBase = FHGrid * KERNEL_FH_K_ELEMS_PER_GRID;

    kValues* kValsTile = kVals + FHGridBase;
    int numElems = MIN(KERNEL_FH_K_ELEMS_PER_GRID, numK - FHGridBase);
    ciErr = clEnqueueWriteBuffer (cq, *c_d, CL_TRUE, 0, numElems * sizeof(kValues), kValsTile, 0, NULL, NULL);
    OclWrapper::checkErr(ciErr, "write buffer c");

    clSetKernelArg(ckFH, 1, sizeof(int), (void*)&FHGridBase);

    ciErr = clEnqueueNDRangeKernel(cq, ckFH, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &evFH[FHGrid]);
    OclWrapper::checkErr(ciErr, "launch FH kernel");
  }
}



int
main (int argc, char *argv[])
{
  int numX, numK;		/* Number of X and K values */
  int original_numK, original_numX;		/* Number of K and X values in input file */
  float *kx, *ky, *kz;		/* K trajectory (3D vectors) */
  float *x, *y, *z;		/* X coordinates (3D vectors) */
  float *phiR, *phiI;		/* Phi values (complex) */
  float *dR, *dI;		/* D values (complex) */
  float *realRhoPhi, *imagRhoPhi;  /* RhoPhi values (complex) */
  float *outI, *outR;		/* Output signal (complex) */
  kValues* kVals;		/* Copy of X and RhoPhi.  Its
				 * data layout has better cache
				 * performance. */

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

  printf("%d pixels in output; %d samples in trajectory; using %d samples\n",
         numX, original_numK, numK);

  // initialize OpenCL context and create command queue
  cl_int ciErr;
  OclWrapper *ocl = new OclWrapper;
  if (!ocl->init (OPENCL_PLATFORM, OPENCL_DEVICE_ID))
    return 1;

  // load and build kernels
  cl_program program;
  if (!ocl->buildProgram ("computeFH.cl", &program))
    return 1;
  cl_kernel ckComputeRhoPhi = clCreateKernel(program, "ComputeRhoPhi", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create kernel RhoPhi")) return 1;
  cl_kernel ckComputeFH = clCreateKernel(program, "ComputeFH", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create kernel FH")) return 1;


  /* Create CPU data structures */
  createDataStructs(numK, numX, realRhoPhi, imagRhoPhi, outR, outI);
  kVals = (kValues*)calloc(numK, sizeof (kValues));

  inf_timer ifGPU;
  startTimer(&ifGPU);

  /* GPU section 1 (precompute Rho, Phi)*/
  {
    /* Mirror several data structures on the device */
    cl_mem phiR_d, phiI_d;
    cl_mem dR_d, dI_d;
    cl_mem realRhoPhi_d, imagRhoPhi_d;

    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numK, sizeof(float), phiR_d, phiR);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numK, sizeof(float), phiI_d, phiI);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numK, sizeof(float), dR_d, dR);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numK, sizeof(float), dI_d, dI);
    realRhoPhi_d = clCreateBuffer(ocl->getContext(), CL_MEM_WRITE_ONLY, numK*sizeof(float), NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create realRhoPhi buffer");
    imagRhoPhi_d = clCreateBuffer(ocl->getContext(), CL_MEM_WRITE_ONLY, numK*sizeof(float), NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create imagRhoPhi buffer");

    /* Pre-compute the values of rhoPhi on the GPU */
    computeRhoPhi_GPU(ocl->getCmdQueue(), ckComputeRhoPhi, argc, argv, numK, &phiR_d, &phiI_d, &dR_d, &dI_d, 
		      &realRhoPhi_d, &imagRhoPhi_d);

    cleanupMemoryGPU(ocl->getCmdQueue(), numK, sizeof(float), realRhoPhi_d, realRhoPhi);
    cleanupMemoryGPU(ocl->getCmdQueue(), numK, sizeof(float), imagRhoPhi_d, imagRhoPhi);
    clReleaseMemObject(phiR_d);
    clReleaseMemObject(phiI_d);
    clReleaseMemObject(dR_d);
    clReleaseMemObject(dI_d);
  }

  clFinish(ocl->getCmdQueue());
  stopTimer(&ifGPU);
  printf ("loop 1: %fms\n", elapsedTime(ifGPU));

  /* Fill in kVals values */
  for (int k = 0; k < numK; k++) {
    kVals[k].Kx = kx[k];
    kVals[k].Ky = ky[k];
    kVals[k].Kz = kz[k];
    kVals[k].RhoPhiR = realRhoPhi[k];
    kVals[k].RhoPhiI = imagRhoPhi[k];
  }

  startTimer(&ifGPU);

  /* GPU section 2 (compute FH)*/
  {
    cl_mem x_d, y_d, z_d;
    cl_mem outI_d, outR_d;
    cl_mem c_d;

    /* Mirror several data structures on the device */
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numX, sizeof(float), x_d, x);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numX, sizeof(float), y_d, y);
    setupMemoryGPU(ocl->getContext(), ocl->getCmdQueue(), numX, sizeof(float), z_d, z);

    // constant buffer
    c_d = clCreateBuffer(ocl->getContext(), CL_MEM_READ_ONLY, sizeof(kValues) * KERNEL_FH_K_ELEMS_PER_GRID, NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create constant buffer");

    // Zero out initial values of outR and outI.
    // GPU veiws these arrays as initialized (cleared) accumulators.
    outR_d = clCreateBuffer(ocl->getContext(), CL_MEM_READ_WRITE, numX*sizeof(float), NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create buffer outR");
    outI_d = clCreateBuffer(ocl->getContext(), CL_MEM_READ_WRITE, numX*sizeof(float), NULL, &ciErr);
    OclWrapper::checkErr(ciErr, "create buffer outI");
    memset(outI, 0, numX*sizeof(float));
    memset(outR, 0, numX*sizeof(float));
    ciErr  = clEnqueueWriteBuffer(ocl->getCmdQueue(), outR_d, CL_TRUE, 0, numX * sizeof(float), outR, 0, NULL, NULL);
    ciErr |= clEnqueueWriteBuffer(ocl->getCmdQueue(), outI_d, CL_TRUE, 0, numX * sizeof(float), outI, 0, NULL, NULL);
    OclWrapper::checkErr(ciErr, "write buffer outI, outR");

    /* Compute FH on the GPU (main computation) */
    computeFH_GPU(ocl->getCmdQueue(), ckComputeFH, argc, argv, numK, numX, &x_d, &y_d, &z_d, kVals, &outR_d, &outI_d, &c_d);

    /* Release memory on GPU */
    cleanupMemoryGPU(ocl->getCmdQueue(), numX, sizeof(float), outR_d, outR);
    cleanupMemoryGPU(ocl->getCmdQueue(), numX, sizeof(float), outI_d, outI);

    clReleaseMemObject(x_d);
    clReleaseMemObject(y_d);
    clReleaseMemObject(z_d);
  }

  clFinish(ocl->getCmdQueue());
  stopTimer(&ifGPU);
  printf ("loop 2: %fms\n", elapsedTime(ifGPU));

  /* Write result to file */
  //outputData(params->outFile, outR, outI, numX);

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
  free (realRhoPhi);
  free (imagRhoPhi);
  free (kVals);
  free (outR);
  free (outI);

#ifdef PROFILING
  // output profiling data
  double diff, avg;
  cl_ulong culStart, culEnd;

  printf ("\n----------------------------\n");

  clGetEventProfilingInfo(evRhoPhi, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
  clGetEventProfilingInfo(evRhoPhi, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);
  diff = (double)(culEnd-culStart)/1000000;
  printf ("RhoPhi: %fms\n", diff);

  avg = 0.0;
  for (int i=0; i<numFH; i++)
  {
    clGetEventProfilingInfo(evFH[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
    clGetEventProfilingInfo(evFH[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);
    diff = (double)(culEnd-culStart)/1000000;
    printf ("FH %d: %fms\n", i, diff);
    
    avg += diff;
  }

  printf ("FH avg: %fms\n", avg/numFH);

#endif

  free(evFH);

  clReleaseKernel(ckComputeFH);
  clReleaseKernel(ckComputeRhoPhi);
  clReleaseProgram(program);
  delete ocl;

  return 0;
}

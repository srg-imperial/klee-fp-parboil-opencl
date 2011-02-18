/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <CL/cl.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "shell.h"

#include <common.h>
#include <ocl-wrapper.h>

#define BLOCK_SIZE 64 // default

#define GRID_SIZE ((1<<16)-1) // originally: 65535=(2^16 - 1)
#define TABLESIZE 2
#define W_MAX_SIZE 10

#define MIN(x,y) (x<y ? x : y)

cl_mem d_Block_Work;
cl_mem d_FinalReduce;
cl_mem d_Output, d_ReductionSum;
cl_mem d_Coors;
cl_mem d_Sprms;
cl_mem d_Wghts;

cl_float4* Coors;
cl_float2* Sprms;

cl_uint4* Block_Work;

int totNumBlocks;
int MaxBlocks;

Atom *ComputeAtom, *BasisAtom;
Shell* ComputeShell;
float Wghts[TABLESIZE];

float *ReductionSum;
int totReductionElements;
int totNumAtoms;
int totBasisShells;

cl_event *evComputeX, *evReduction;

cl_uint2 *FinalReduce;

void AllocateDataOnDevice(cl_context, cl_command_queue, int, int, int, int, int, int);
void RunKernel(int, int, cl_command_queue, cl_kernel, cl_kernel);

void CalcOnHost(int);
Atom* ReadBasisAtoms(int&, const char *);
int TotalNumOfShells(char*, int, int&);
void PopulateShells(char*, int);
void PopulateHostData(int, int, int);
int NumOfIntegrals(int, int);
void DistributeBlockWork(int, int);
void FreeAllData();
void PopulateWeights();

double root1(double X);

void printProfiling();

int main(int argc, char* argv[])
{
  totNumBlocks = 0;
  MaxBlocks = 0;

  int blockSize = BLOCK_SIZE; // default
  getCmdLineParamInt("-local", argc, argv, (int*)&blockSize);

  char *inputfile = argv[1];
  char *inputfile2 = argv[2];

  int numBasisAtoms;
  BasisAtom = ReadBasisAtoms(numBasisAtoms, inputfile2);
  
  int totNumShells = TotalNumOfShells(inputfile, numBasisAtoms, 
                                        totNumAtoms);

  totReductionElements = totNumShells * (totNumShells + 1) * 
    (totNumShells + 2) * (totNumShells + 3) / 24;
  ComputeAtom = (Atom*)malloc(totNumAtoms * sizeof(Atom));
  ComputeShell = (Shell*)malloc(totNumShells * sizeof(Shell));

  PopulateShells(inputfile, numBasisAtoms);
  //        all shells are ready now

  //        prepare host data
  totBasisShells = 0;
  for(int i = 0; i < numBasisAtoms; i ++) 
    for(int j = 0; j < BasisAtom[i].numShells; j ++)
      totBasisShells += BasisAtom[i].AtomShell[j].numPrimitives;

  Coors = (cl_float4*)malloc(totNumAtoms * sizeof(cl_float4));
  Sprms = (cl_float2*)malloc(totBasisShells * sizeof(cl_float2));
  PopulateHostData(totNumAtoms, totNumShells, numBasisAtoms);

  //        distribute the work now
  FinalReduce = (cl_uint2*)malloc(totReductionElements * sizeof(cl_uint2));
  int numIntegrals = NumOfIntegrals(totNumShells, blockSize);
  printf("Total # of integrals to compute: %d\n", numIntegrals);
  printf("Total # of blocks allocated: %d\n", totNumBlocks);
  printf("Final array size: %d\n", totReductionElements);
  Block_Work = (cl_uint4*)malloc(totNumBlocks * sizeof(cl_uint4));
  DistributeBlockWork(totNumShells, blockSize);

  int d_output_mem = totNumBlocks * sizeof(float);
  int d_work_mem = totNumBlocks * sizeof(cl_uint4);
  int reduction_mem = totReductionElements * sizeof(float);
  int final_mem = totReductionElements * sizeof(cl_uint2);

  ReductionSum = (float*)malloc(reduction_mem);

  // initialize OpenCL context and create command queue
  cl_int ciErr;
  OclWrapper *ocl = new OclWrapper;
  if (!ocl->init (OPENCL_PLATFORM, OPENCL_DEVICE_ID))
    return 1;

  // load and build kernel
  cl_program program;
  if (!ocl->buildProgram ("crys_kernel.cl", &program))
    return 1;
  cl_kernel ckComputeX = clCreateKernel(program, "ComputeX", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create computeX kernel")) return 1;
  cl_kernel ckReduction = clCreateKernel(program, "DoReduction", &ciErr);
  if (!OclWrapper::checkErr(ciErr, "create reduction kernel")) return 1;

  // prepare device data
  AllocateDataOnDevice(ocl->getContext(), ocl->getCmdQueue(),
                       d_output_mem, d_work_mem, reduction_mem, 
                       final_mem, totNumAtoms, totBasisShells);
  int d_total_mem = d_output_mem + d_work_mem + reduction_mem + final_mem;
  printf("%.2lf MB allocated\n", (double)d_total_mem / 1048576);
  printf("maxblocks = %d\n", MaxBlocks);


  // okay, now ready to do something useful
  RunKernel(numIntegrals, blockSize, ocl->getCmdQueue(), ckComputeX, ckReduction);

  // loading data back to the host
  ciErr = clEnqueueReadBuffer (ocl->getCmdQueue(), d_ReductionSum, CL_TRUE, 0, reduction_mem, ReductionSum, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "read buffer: ReductionSum");

 // REMOVE
  float *output = (float*)malloc (totNumBlocks * sizeof(float));
  ciErr = clEnqueueReadBuffer (ocl->getCmdQueue(), d_Output, CL_TRUE, 0, totNumBlocks * sizeof(float), output, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "read buffer: output");
 // REMOVE

#ifdef PROFILING
  printProfiling();
#endif

  free (evComputeX);
  free (evReduction);

  FreeAllData();

  delete ocl;

  return 0;
}

void RunKernel(int numIntegrals, int blockSize, cl_command_queue cmdQueue, cl_kernel ckComputeX, cl_kernel ckReduction)
{
  cl_int ciErr;
  int runs = (int)(ceil(1.0 * totNumBlocks / GRID_SIZE));
  printf("%d computation cycles will be performed...\n", runs);
  int RemainingBlocks = totNumBlocks;
  int StartBlock = 0;

  evComputeX = (cl_event*) malloc (sizeof(cl_event) * runs);

#ifdef PROFILING
  // profile kernel executions
  inf_timer tComputeX;
  startTimer(&tComputeX);
#endif

  clSetKernelArg (ckComputeX, 0, sizeof(cl_mem), &d_Block_Work);
  clSetKernelArg (ckComputeX, 1, sizeof(cl_mem), &d_Output);
  clSetKernelArg (ckComputeX, 3, sizeof(cl_mem), &d_Coors);
  clSetKernelArg (ckComputeX, 4, sizeof(cl_mem), &d_Sprms);
  clSetKernelArg (ckComputeX, 5, sizeof(float)*blockSize, 0);

  for(int run = 0; run < runs; run ++)
    {
      int numBlocks = MIN(GRID_SIZE, RemainingBlocks);
      size_t localWorkSize = blockSize;
      size_t globalWorkSize = numBlocks*blockSize;

      clSetKernelArg (ckComputeX, 2, sizeof(int), (void*)&StartBlock);

      ciErr = clEnqueueNDRangeKernel(cmdQueue, ckComputeX, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &evComputeX[run]);
      if (!OclWrapper::checkErr (ciErr, "launch computeX kernel")) return;

      RemainingBlocks -= GRID_SIZE;
      StartBlock += numBlocks;
    }


#ifdef PROFILING
  clFinish(cmdQueue);
  stopTimer(&tComputeX);
  printf ("compute x: %10fms\n", elapsedTime(tComputeX));
#endif

  runs = (int)(ceil(1.0 * totReductionElements / GRID_SIZE));
  printf("done.\n\n%d reduction cycles will be performed...\n", runs);
  int RemainReduction = totReductionElements;
  int Offset = 0;

  evReduction = (cl_event*) malloc (sizeof(cl_event) * runs);

#ifdef PROFILING
  // profile kernel executions
  inf_timer tReduction;
  startTimer(&tReduction);
#endif

  clSetKernelArg (ckReduction, 0, sizeof(cl_mem), &d_ReductionSum);
  clSetKernelArg (ckReduction, 1, sizeof(cl_mem), &d_Output);
  clSetKernelArg (ckReduction, 2, sizeof(cl_mem), &d_FinalReduce);
  clSetKernelArg (ckReduction, 4, sizeof(float) * blockSize, 0);

  for(int run = 0; run < runs; run ++)
    {
      //int numBlocks = MIN(GRID_SIZE, RemainReduction);
      int numBlocks = MIN(GRID_SIZE, RemainReduction);

      size_t localWorkSize = MaxBlocks;
      size_t globalWorkSize = numBlocks*MaxBlocks;

      clSetKernelArg (ckReduction, 3, sizeof(int), (void*)&Offset);

      ciErr = clEnqueueNDRangeKernel(cmdQueue, ckReduction, 1, NULL, &globalWorkSize, &localWorkSize, 0, NULL, &evReduction[run]);
      if (!OclWrapper::checkErr (ciErr, "launch reduction kernel")) return;

      RemainReduction -= GRID_SIZE;
      Offset += numBlocks;
    }

#ifdef PROFILING
  clFinish(cmdQueue);
  stopTimer(&tReduction);
  printf ("reduction: %10fms\n", elapsedTime(tReduction));
#endif

}


void AllocateDataOnDevice(cl_context ctxt, cl_command_queue cmdQueue,
                          int d_output_mem, int d_work_mem, 
                          int reduction_mem, int final_mem,
                          int numCoors, int numSprms)
{
  cl_int ciErr;

  d_ReductionSum = clCreateBuffer(ctxt, CL_MEM_WRITE_ONLY, reduction_mem, NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer: ReductionSum");
  d_Output = clCreateBuffer(ctxt, CL_MEM_READ_WRITE, d_output_mem, NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer: Output");
  d_Block_Work = clCreateBuffer(ctxt, CL_MEM_READ_ONLY, d_work_mem, NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer: Block Work");
  d_FinalReduce = clCreateBuffer(ctxt, CL_MEM_READ_ONLY, final_mem, NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer: FinalReduce");
  d_Coors = clCreateBuffer(ctxt, CL_MEM_READ_ONLY, totNumAtoms*sizeof(cl_float4), NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer: Coors");
  d_Sprms = clCreateBuffer(ctxt, CL_MEM_READ_ONLY, totBasisShells*sizeof(cl_float2), NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer: Sprms");
  d_Wghts = clCreateBuffer(ctxt, CL_MEM_READ_ONLY, TABLESIZE*sizeof(float), NULL, &ciErr);
  OclWrapper::checkErr(ciErr, "create buffer: Wghts");
  
  ciErr = clEnqueueWriteBuffer (cmdQueue, d_Block_Work, CL_TRUE, 0, d_work_mem, Block_Work, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "copy buffer: Block Work");
  ciErr = clEnqueueWriteBuffer (cmdQueue, d_FinalReduce, CL_TRUE, 0, final_mem, FinalReduce, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "copy buffer: FinalReduce");
  ciErr = clEnqueueWriteBuffer (cmdQueue, d_Coors, CL_TRUE, 0, totNumAtoms * sizeof(cl_float4), Coors, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "copy buffer: Coors");
  ciErr = clEnqueueWriteBuffer (cmdQueue, d_Sprms, CL_TRUE, 0, totBasisShells * sizeof(cl_float2), Sprms, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "copy buffer: Sprms");
  ciErr = clEnqueueWriteBuffer (cmdQueue, d_Wghts, CL_TRUE, 0, TABLESIZE * sizeof(float), Wghts, 0, NULL, NULL);
  OclWrapper::checkErr(ciErr, "copy buffer: Wghts");
}

void FreeAllData()
{
  clReleaseMemObject(d_FinalReduce);
  clReleaseMemObject(d_Block_Work);
  clReleaseMemObject(d_Output);
  clReleaseMemObject(d_ReductionSum);
  clReleaseMemObject(d_Coors);
  clReleaseMemObject(d_Wghts);
  clReleaseMemObject(d_Sprms);
  
  free ((void*)Block_Work);
  free ((void*)FinalReduce);
  
  free ((void*)ComputeAtom);
  free ((void*)BasisAtom);
  free ((void*)ComputeShell);
}

Atom* ReadBasisAtoms(int& numBasisAtoms, const char* filename)
{
  FILE* basis = fopen(filename, "r");
  if(!basis)
    {
      printf("Unable to open file %s\n", filename);
      exit(0);
    }
  int numAtoms = 0, numShells = 0;
  fscanf(basis, "%*s %*s %d", &numAtoms);
  fscanf(basis, "%*s %*s %d", &numShells);
  //printf("\n>>>>>>> STARTED BASIS SET OUTPUT <<<<<<<\n");
  //printf("\n# OF KNOWN ATOMS:  %d\n", numAtoms);
  //printf("# OF KNOWN SHELLS: %d\n\n", numShells);
  numBasisAtoms = numAtoms;
  
  Atom* BasisAtom = (Atom*)malloc(numAtoms * sizeof(Atom));
  
  for(int atom = 0; atom < numAtoms; atom ++)
    {
      char type[4];
      char buff[4];
      fscanf(basis, "%*s %s", type);
      fscanf(basis, "%*s %d", &numShells);
      BasisAtom[atom].numShells = numShells;
      strcpy(BasisAtom[atom].Type, type);
      //printf("\nAtom %s (%d shells)\n", BasisAtom[atom].Type, 
             //BasisAtom[atom].numShells);
      
      for(int shell = 0; shell < numShells; shell ++)
        {
          int numPrimitives = 0;
          fscanf(basis, "%*s %*d %*s %d", &numPrimitives);
          BasisAtom[atom].AtomShell[shell].numPrimitives = numPrimitives;
          sprintf(buff, "%d", shell + 1);
          strcpy(BasisAtom[atom].AtomShell[shell].Type    , type);
          strcpy(BasisAtom[atom].AtomShell[shell].Type + 1, buff);
          //printf("\tShell %s: %d primitives\n", 
                 //BasisAtom[atom].AtomShell[shell].Type,
                 //BasisAtom[atom].AtomShell[shell].numPrimitives);
          for(int prim = 0; prim < numPrimitives; prim ++)
            {
              fscanf(basis, "%*s %*s %*s %f %f", 
                     &BasisAtom[atom].AtomShell[shell].Alpha[prim], 
                     &BasisAtom[atom].AtomShell[shell].Coeff[prim]);
              //printf("\t\tprimitive %d: %10.2f    %5.2f\n", prim + 1, 
                     //BasisAtom[atom].AtomShell[shell].Alpha[prim], 
                     //BasisAtom[atom].AtomShell[shell].Coeff[prim]);
            }
          //printf("\n");
        }
    }
  //printf(">>>>>>>> DONE BASIS SET OUTPUT <<<<<<<<\n\n\n");
  fclose(basis);
  return BasisAtom;
}

int TotalNumOfShells(char* fname, int numBasisAtoms, int& totNumAtoms)
{
  FILE* inp = fopen(fname, "r");
  if(!inp)
    {
      printf("Unable to open %s\n", fname);
      exit(0);
    }
  int numShells = 0;
  fscanf(inp, "%*s %d", &totNumAtoms);
  
  for(int atom = 0; atom < totNumAtoms; atom ++)
    {
      char type[8];
      fscanf(inp, "%s %*s %*s %*s", type);
      
      int notfound = 1;
      for(int batom = 0; batom < numBasisAtoms; batom ++)
        {
          if(!strcmp(BasisAtom[batom].Type, type))
            {
              numShells += BasisAtom[batom].numShells;
              notfound = 0;
              break;
            }
        }
      if(notfound)
        {
          printf("Unable to find atom \'%s\' in the basis set\n", type);
          exit(0);
        }
    }

  fclose(inp);
  return numShells;
}

void PopulateShells(char* fname, int numBasisAtoms)
{
  FILE* inp = fopen(fname, "r");
  if(!inp)
    {
      printf("Unable to open %s\n", fname);
      exit(0);
    }
  int numAtoms = 0, currentShell = 0;
  fscanf(inp, "%*s %d", &numAtoms);
  
  for(int atom = 0; atom < numAtoms; atom ++)
    {
      fscanf(inp, "%s %f %f %f", &ComputeAtom[atom].Type, 
             &ComputeAtom[atom].X,
             &ComputeAtom[atom].Y, &ComputeAtom[atom].Z);
      
      int currentInList = 0;
      for(int batom = 0; batom < numBasisAtoms; batom ++)
        {
          if(!strcmp(BasisAtom[batom].Type, ComputeAtom[atom].Type))
            {
              for(int shell = 0; shell < BasisAtom[batom].numShells; 
                  shell ++)
                {
                  ComputeShell[currentShell] = 
                    BasisAtom[batom].AtomShell[shell];
                  ComputeShell[currentShell].myAtom = atom;
                  
                  //        this part populates inList
                  for(int prim = 0; prim < 
                        BasisAtom[batom].AtomShell[shell].numPrimitives; 
                      prim ++)
                    ComputeShell[currentShell].inList[prim] = 
                      currentInList ++;

                  currentShell ++;
                }
              break;
            }
          //        this part populates inList
          else
            {
              for(int shell = 0; shell < BasisAtom[batom].numShells; 
                  shell ++)
                currentInList += 
                  BasisAtom[batom].AtomShell[shell].numPrimitives;
            }
        }
    }
  fclose(inp);
}

void PopulateHostData(int totNumAtoms, int totNumShells, int numBasisAtoms)
{
  PopulateWeights();
  for(int atom = 0; atom < totNumAtoms; atom ++)
    {
      Coors[atom].s[0] = ComputeAtom[atom].X;
      Coors[atom].s[1] = ComputeAtom[atom].Y;
      Coors[atom].s[2] = ComputeAtom[atom].Z;
    }

  int currentPos = 0;
  for(int batom = 0; batom < numBasisAtoms; batom ++)
    {
      for(int shell = 0; shell < BasisAtom[batom].numShells; shell ++)
        {
          for(int prim = 0; prim < 
                BasisAtom[batom].AtomShell[shell].numPrimitives; 
              prim ++)
            {
              Sprms[currentPos].s[0] = 
                BasisAtom[batom].AtomShell[shell].Alpha[prim];
              Sprms[currentPos].s[1] = 
                BasisAtom[batom].AtomShell[shell].Coeff[prim];
              currentPos ++;
            }
        }
    }
}

int NumOfIntegrals(int totNumShells, int blockSize)
{
  int numIntegrals = 0;
  int firstRedElement = 0;
  int redElement = 0;
  for(int shell1 = 0; shell1 < totNumShells; shell1 ++)
    for(int shell2 = shell1; shell2 < totNumShells; shell2 ++)
      for(int shell3 = shell2; shell3 < totNumShells; shell3 ++)
        for(int shell4 = shell3; shell4 < totNumShells; shell4 ++)
          {
            int integrals = ComputeShell[shell1].numPrimitives * 
              ComputeShell[shell2].numPrimitives * 
              ComputeShell[shell3].numPrimitives * 
              ComputeShell[shell4].numPrimitives;
            numIntegrals += integrals;
            
            int blocks = (int)ceil(1.0 * integrals / blockSize);
            totNumBlocks += blocks;
            if(blocks > MaxBlocks)
              MaxBlocks = blocks;
            FinalReduce[redElement].s[0] = firstRedElement;
            FinalReduce[redElement].s[1] = blocks;
            firstRedElement += blocks;
            redElement ++;
          }
  return numIntegrals;
}

void DistributeBlockWork(int totNumShells, int blockSize)
{
  int numElements = 0;
  int StartBlock = 0;
  for(int shell1 = 0; shell1 < totNumShells; shell1 ++)
    {
      for(int shell2 = shell1; shell2 < totNumShells; shell2 ++)
        {
          for(int shell3 = shell2; shell3 < totNumShells; shell3 ++)
            {
              for(int shell4 = shell3; shell4 < totNumShells; shell4 ++)
                {
                  int integrals = ComputeShell[shell1].numPrimitives * 
                    ComputeShell[shell2].numPrimitives * 
                    ComputeShell[shell3].numPrimitives * 
                    ComputeShell[shell4].numPrimitives;

                  int blocks = (int)ceil(1.0 * integrals / blockSize);
                  StartBlock = numElements;
                  
                  for(int block = 0; block < blocks; block ++)
                    {
                      int a4 = ComputeShell[shell4].numPrimitives;
                      int a3 = ComputeShell[shell3].numPrimitives;
                      int a2 = ComputeShell[shell2].numPrimitives;
                      int a1 = ComputeShell[shell1].numPrimitives;
                      
                      int offset4 = ComputeShell[shell4].inList[0];
                      int offset3 = ComputeShell[shell3].inList[0];
                      int offset2 = ComputeShell[shell2].inList[0];
                      int offset1 = ComputeShell[shell1].inList[0];
                      
                      Block_Work[numElements].s[1] = 
                        (ComputeShell[shell1].myAtom << 24) | 
                        (ComputeShell[shell2].myAtom << 16) | 
                        (ComputeShell[shell3].myAtom << 8 ) | 
                        (ComputeShell[shell4].myAtom      ) ;
                      
                      Block_Work[numElements].s[2] = 
                        (offset1 << 24) | 
                        (offset2 << 16) | 
                        (offset3 << 8 ) | 
                        (offset4) ;
                      
                      Block_Work[numElements].s[0] = 
                        (a1         << 12) |
                        (a2         <<  8) |
                        (a3         <<  4) |
                        (a4              ) ;
                      
                      Block_Work[numElements].s[3] = StartBlock;
                      
                      numElements ++;
                    }
                }
            }
        }
    }
}

double root1(double X)
{
  double PIE4;
  double WW1 = 0.0;
  double F1,E,Y,inv;
  
  PIE4 = 7.85398163397448E-01;
  
  if (X < 3.0e-7)
    {
      WW1 = 1.0 - 0.333333333 * X;
    } 
  else if (X < 1.0) 
    {
      F1 = ((((((((-8.36313918003957E-08*X+1.21222603512827E-06 )*X-
                  1.15662609053481E-05 )*X+9.25197374512647E-05 )*X-
                6.40994113129432E-04 )*X+3.78787044215009E-03 )*X-
              1.85185172458485E-02 )*X+7.14285713298222E-02 )*X-
            1.99999999997023E-01 )*X+3.33333333333318E-01;
      WW1 = (X+X)*F1 + exp(-X);
    } 
  else if (X < 3.0) 
    {
      Y = X-2.0;
      F1 = ((((((((((-1.61702782425558E-10*Y+1.96215250865776E-09 )*Y-
                    2.14234468198419E-08 )*Y+2.17216556336318E-07 )*Y-
                  1.98850171329371E-06 )*Y+1.62429321438911E-05 )*Y-
                1.16740298039895E-04 )*Y+7.24888732052332E-04 )*Y-
              3.79490003707156E-03 )*Y+1.61723488664661E-02 )*Y-
            5.29428148329736E-02 )*Y+1.15702180856167E-01;
      WW1 = (X+X)*F1+exp(-X);
      
    } 
  else if (X < 5.0)
    {
      Y = X-4.0;
      F1 = ((((((((((-2.62453564772299E-11*Y+3.24031041623823E-10 )*Y-
                    3.614965656163E-09)*Y+3.760256799971E-08)*Y-
                  3.553558319675E-07)*Y+3.022556449731E-06)*Y-
                2.290098979647E-05)*Y+1.526537461148E-04)*Y-
              8.81947375894379E-04)*Y+4.33207949514611E-03 )*Y-
            1.75257821619926E-02 )*Y+5.28406320615584E-02;
      WW1 = (X+X)*F1+exp(-X);
      
    } 
  else if (X < 10.0) 
    {
      E = exp(-X);
      inv = 1 / X;
      WW1 = (((((( 4.6897511375022E-01*inv-6.9955602298985E-01)*inv +
                 5.3689283271887E-01)*inv-3.2883030418398E-01)*inv +
               2.4645596956002E-01)*inv-4.9984072848436E-01)*inv -
             3.1501078774085E-06)*E + sqrt(PIE4*inv);
      
    } 
  else if (X < 15.0) 
    {
      E = exp(-X);
      inv = 1 / X;
      WW1 = (((-1.8784686463512E-01*inv+2.2991849164985E-01)*inv -
              4.9893752514047E-01)*inv-2.1916512131607E-05)*E \
        + sqrt(PIE4*inv);
      
    } 
  else if (X < 33.0) 
    {
      E = exp(-X);
      inv = 1 / X;
      WW1 = (( 1.9623264149430E-01*inv-4.9695241464490E-01)*inv -
             6.0156581186481E-05)*E + sqrt(PIE4*inv);

    } 
  else 
    {
      inv = 1 / X;
      WW1 = sqrt(PIE4*inv);
    }
  
  return WW1;
}

void PopulateWeights()
{
  for(int i = 0; i < TABLESIZE; i ++)
    {
      float X = (float)(1.0 * i * W_MAX_SIZE / (TABLESIZE - 1));
      Wghts[i] = root1(X);
    }
}

void printProfiling()
{
  printf ("\n-------- PROFILING RESULTS --------\n");
  // output kernel runtimes
  double avg, time;
  cl_ulong culStart, culEnd;
  int i;

  int runs = (int)(ceil(1.0 * totNumBlocks / GRID_SIZE));

  for (i=0; i<runs; i++)
  {
    clGetEventProfilingInfo(evComputeX[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
    clGetEventProfilingInfo(evComputeX[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);

    time = (double)(culEnd-culStart)/1000000;
    printf ("ComputeX kernel %d: %fms\n", i, time);

    avg += time;
  }
  printf ("\n\tComputeX kernel avg: %fms\n\n", avg/runs);


  runs = (int)(ceil(1.0 * totReductionElements / GRID_SIZE));
  avg = 0;
  for (int i=0; i<runs; i++)
  {
    clGetEventProfilingInfo(evReduction[i], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &culStart, NULL);
    clGetEventProfilingInfo(evReduction[i], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &culEnd, NULL);

    time = (double)(culEnd-culStart)/1000000;
    printf ("reduction kernel %d: %fms\n", i, time);

    avg += time;
  }
  printf ("\n\treduction kernel avg: %fms\n\n", avg/runs);

}

/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/
#include <stdio.h>
#include <string.h>
#include "shell.h"

#include <common.h>


#define CUDA_ERRCK { cudaError_t err; \
  if ((err = cudaGetLastError()) != cudaSuccess) { \
  printf("CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
  exit (-1); }}


#define BLOCK_SIZE 64


uint4* d_Block_Work;
uint2* d_FinalReduce;
float *d_Output, *d_ReductionSum;
float4 *d_Coors;
float2 *d_Sprms;
float *d_Wghts;

#include "crys_kernel.cu"

float4* Coors;
float2* Sprms;

uint4* Block_Work;

int totNumBlocks;
int MaxBlocks;

Atom *ComputeAtom, *BasisAtom;
Shell* ComputeShell;
float Wghts[TABLESIZE];

float *ReductionSum;
int totReductionElements;
int totNumAtoms;
int totBasisShells;

uint2 *FinalReduce;

struct double3
{
  double x, y, z;
};

void AllocateDataOnDevice(int, int, int, int, int, int);
void RunKernel(int, int);

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
  //	all shells are ready now

  //	prepare host data
  totBasisShells = 0;
  for(int i = 0; i < numBasisAtoms; i ++) 
    for(int j = 0; j < BasisAtom[i].numShells; j ++)
      totBasisShells += BasisAtom[i].AtomShell[j].numPrimitives;

  Coors = (float4*)malloc(totNumAtoms * sizeof(float4));
  Sprms = (float2*)malloc(totBasisShells * sizeof(float2));
  PopulateHostData(totNumAtoms, totNumShells, numBasisAtoms);

  //	distribute the work now
  FinalReduce = (uint2*)malloc(totReductionElements * sizeof(uint2));
  int numIntegrals = NumOfIntegrals(totNumShells, blockSize);
  printf("Total # of integrals to compute: %d\n", numIntegrals);
  printf("Total # of blocks allocated: %d\n", totNumBlocks);
  printf("Final array size: %d\n", totReductionElements);
  Block_Work = (uint4*)malloc(totNumBlocks * sizeof(uint4));
  DistributeBlockWork(totNumShells, blockSize);

  int d_output_mem = totNumBlocks * sizeof(float);
  int d_work_mem = totNumBlocks * sizeof(uint4);
  int reduction_mem = totReductionElements * sizeof(float);
  int final_mem = totReductionElements * sizeof(uint2);

  ReductionSum = (float*)malloc(reduction_mem);

  //	prepare device data
  AllocateDataOnDevice(d_output_mem, d_work_mem, reduction_mem, 
		       final_mem, totNumAtoms, totBasisShells);
  int d_total_mem = d_output_mem + d_work_mem + reduction_mem + final_mem;
  printf("%.2lf MB allocated\n", (double)d_total_mem / 1048576);
  printf("maxblocks = %d\n", MaxBlocks);

  //	okay, now ready to do something useful
  RunKernel(numIntegrals, blockSize);

  //	loading data back to the host
  cudaMemcpy(ReductionSum, d_ReductionSum, 
	     reduction_mem, cudaMemcpyDeviceToHost);
  CUDA_ERRCK

  // print some of the output data
  int count=0;
  for (int i=0; i<totReductionElements; i++)
    if (ReductionSum[i] != 0.0)
      count++;
  printf ("reduction count: %d (out of %d)\n", count, totReductionElements);

  FreeAllData();

  return 0;
}

void RunKernel(int numIntegrals, int blockSize)
{
  int runs = (int)(ceil(1.0 * totNumBlocks / GRID_SIZE));
  printf("%d computation cycles will be performed...\n", runs);
  int RemainingBlocks = totNumBlocks;
  int StartBlock = 0;

#ifdef PROFILING
  // profile kernel executions
  cudaEvent_t *evStart = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * runs);
  cudaEvent_t *evStop  = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * runs);
  for (int i=0; i<runs; i++)
  {
    cudaEventCreate(&evStart[i]);
    cudaEventCreate(&evStop[i]);
  }

  inf_timer tComputeX;
  startTimer(&tComputeX);
#endif

  for(int run = 0; run < runs; run ++)
    {
      int numBlocks = min(GRID_SIZE, RemainingBlocks);
      dim3 grid(numBlocks, 1, 1);
      dim3 block(blockSize, 1, 1);

#ifdef PROFILING
      cudaEventRecord(evStart[run], 0); 
#endif

      ComputeX <<< grid, block, blockSize*sizeof(float) >>> (d_Block_Work, d_Output, StartBlock, d_Coors,
                                    d_Sprms);
      CUDA_ERRCK
      //if (params->synchronizeGpu) cudaThreadSynchronize();

#ifdef PROFILING
      cudaEventRecord(evStop[run], 0); 
#endif

      RemainingBlocks -= GRID_SIZE;
      StartBlock += numBlocks;
    }


#ifdef PROFILING
  cudaThreadSynchronize();
  stopTimer(&tComputeX);
  printf ("compute x: %10fms\n", elapsedTime(tComputeX));

  // output individual kernel runtimes
  float avg;
  for (int i=0; i<runs; i++)
  {
    float diff;
    cudaEventElapsedTime(&diff, evStart[i], evStop[i]);
    printf ("ComputeX kernel %d: %fms\n", i, diff);

    avg += diff;
  }
  printf ("\n\tComputeX kernel avg: %fms\n\n", avg/runs);

  free(evStart);
  free(evStop);
#endif


#ifdef PROFILING
  // profile kernel executions
  evStart = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * runs);
  evStop  = (cudaEvent_t*) malloc (sizeof(cudaEvent_t) * runs);
  for (int i=0; i<runs; i++)
  {
    cudaEventCreate(&evStart[i]);
    cudaEventCreate(&evStop[i]);
  }

  inf_timer tReduction;
  startTimer(&tReduction);
#endif

  runs = (int)(ceil(1.0 * totReductionElements / GRID_SIZE));
  printf("done.\n\n%d reduction cycles will be performed...\n", runs);
  int RemainReduction = totReductionElements;
  int Offset = 0;
  for(int run = 0; run < runs; run ++)
    {
      int numBlocks = min(GRID_SIZE, RemainReduction);

      dim3 grid(numBlocks, 1, 1);
      dim3 block(MaxBlocks, 1, 1);

#ifdef PROFILING
      cudaEventRecord(evStart[run], 0); 
#endif

      DoReduction <<< grid, block, blockSize*sizeof(float) >>> (d_ReductionSum, d_Output, 
				       d_FinalReduce, MaxBlocks, Offset);
      CUDA_ERRCK
      //if (params->synchronizeGpu) cudaThreadSynchronize();

#ifdef PROFILING
      cudaEventRecord(evStop[run], 0); 
#endif

      RemainReduction -= GRID_SIZE;
      Offset += numBlocks;
    }

#ifdef PROFILING
  cudaThreadSynchronize();
  stopTimer(&tReduction);
  printf ("reduction: %10fms\n", elapsedTime(tReduction));

  // output individual kernel runtimes
  avg = 0;
  for (int i=0; i<runs; i++)
  {
    float diff;
    cudaEventElapsedTime(&diff, evStart[i], evStop[i]);
    printf ("reduction kernel %d: %fms\n", i, diff);

    avg += diff;
  }
  printf ("\n\treduction kernel avg: %fms\n\n", avg/runs);

  free(evStart);
  free(evStop);
#endif

}


void AllocateDataOnDevice(int d_output_mem, int d_work_mem, 
			  int reduction_mem, int final_mem,
			  int numCoors, int numSprms)
{
  cudaMalloc((void**)&d_ReductionSum, reduction_mem);
  CUDA_ERRCK
  cudaMalloc((void**)&d_Output, d_output_mem);
  CUDA_ERRCK
  cudaMalloc((void**)&d_Block_Work, d_work_mem);
  CUDA_ERRCK
  cudaMalloc((void**)&d_FinalReduce, final_mem);
  CUDA_ERRCK
  cudaMalloc((void**)&d_Coors, totNumAtoms * sizeof(float4));
  CUDA_ERRCK
  cudaMalloc((void**)&d_Sprms, totBasisShells * sizeof(float2));
  CUDA_ERRCK
  cudaMalloc((void**)&d_Wghts, TABLESIZE*sizeof(float));
  CUDA_ERRCK
  
  cudaMemcpy(d_Block_Work, Block_Work, d_work_mem, 
	     cudaMemcpyHostToDevice);
  CUDA_ERRCK
  cudaMemcpy(d_FinalReduce, FinalReduce, final_mem, 
	     cudaMemcpyHostToDevice);
  CUDA_ERRCK
  cudaMemcpy(d_Coors, Coors, totNumAtoms * sizeof(float4),
  	     cudaMemcpyHostToDevice);
  CUDA_ERRCK
  cudaMemcpy(d_Sprms, Sprms, totBasisShells * sizeof(float2),
  	     cudaMemcpyHostToDevice);
  CUDA_ERRCK
  cudaMemcpy(d_Wghts, Wghts, TABLESIZE*sizeof(float),
  	     cudaMemcpyHostToDevice);
  CUDA_ERRCK
}

void FreeAllData()
{
  cudaFree((void*)d_FinalReduce);
  CUDA_ERRCK
  cudaFree((void*)d_Block_Work);
  CUDA_ERRCK
  cudaFree((void*)d_Output);
  CUDA_ERRCK
  cudaFree((void*)d_ReductionSum);
  CUDA_ERRCK
  cudaFree(d_Coors);
  CUDA_ERRCK
  cudaFree(d_Wghts);
  CUDA_ERRCK
  cudaFree(d_Sprms);
  CUDA_ERRCK
  
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
		  
		  //	this part populates inList
		  for(int prim = 0; prim < 
			BasisAtom[batom].AtomShell[shell].numPrimitives; 
		      prim ++)
		    ComputeShell[currentShell].inList[prim] = 
		      currentInList ++;

		  currentShell ++;
		}
	      break;
	    }
	  //	this part populates inList
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
      Coors[atom].x = ComputeAtom[atom].X;
      Coors[atom].y = ComputeAtom[atom].Y;
      Coors[atom].z = ComputeAtom[atom].Z;
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
	      Sprms[currentPos].x = 
		BasisAtom[batom].AtomShell[shell].Alpha[prim];
	      Sprms[currentPos].y = 
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
	    FinalReduce[redElement].x = firstRedElement;
	    FinalReduce[redElement].y = blocks;
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
		      
		      Block_Work[numElements].y = 
			(ComputeShell[shell1].myAtom << 24) | 
			(ComputeShell[shell2].myAtom << 16) | 
			(ComputeShell[shell3].myAtom << 8 ) | 
			(ComputeShell[shell4].myAtom      ) ;
		      
		      Block_Work[numElements].z = 
			(offset1 << 24) | 
			(offset2 << 16) | 
			(offset3 << 8 ) | 
			(offset4) ;
		      
		      Block_Work[numElements].x = 
			(a1         << 12) |
			(a2         <<  8) |
			(a3         <<  4) |
			(a4              ) ;
		      
		      Block_Work[numElements].w = StartBlock;
		      
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

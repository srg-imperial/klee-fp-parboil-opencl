/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

#define PI   3.1415926535897932384626433832795029f
#define PIx2 6.2831853071795864769252867665590058f

/* Adjustable parameters */
#define KERNEL_FH_K_ELEMS_PER_GRID 512

struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float RhoPhiR;
  float RhoPhiI;
};

__kernel void
ComputeRhoPhi(int numK,
                 __global float* phiR, __global float* phiI, 
                 __global float* dR, __global float* dI, 
                 __global float* realRhoPhi, __global float* imagRhoPhi)
{
  int indexK = get_global_id(0);
  if (indexK < numK) {
    float rPhiR = phiR[indexK];
    float rPhiI = phiI[indexK];
    float rDR = dR[indexK];
    float rDI = dI[indexK];
    realRhoPhi[indexK] = rPhiR * rDR + rPhiI * rDI;
    imagRhoPhi[indexK] = rPhiR * rDI - rPhiI * rDR;
  }
}

__kernel void
ComputeFH(int numK, int kGlobalIndex,
              __global float* x, __global float* y, __global float* z, 
              __global float* outR, __global float* outI,
              __constant struct kValues* c, int numX)
{
  float sX;
  float sY;
  float sZ;
  float sOutR;
  float sOutI;

  // Determine the element of the X arrays computed by this thread
  int xIndex = get_global_id(0);

  if (xIndex < numX) {
    sX = x[xIndex];
    sY = y[xIndex];
    sZ = z[xIndex];
    sOutR = outR[xIndex];
    sOutI = outI[xIndex];

    // Loop over all elements of K in constant mem to compute a partial value
    // for X.
    int kIndex = 0;
    int kCnt = numK - kGlobalIndex;
    if (kCnt < KERNEL_FH_K_ELEMS_PER_GRID) {
      for (kIndex = 0;
           (kIndex < (kCnt % 4)) && (kGlobalIndex < numK);
           kIndex++, kGlobalIndex++) {
        float expArg = PIx2 *
          (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
        float cosArg = native_cos(expArg);
        float sinArg = native_sin(expArg);
        sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
        sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
      }
    }

    for (;
         (kIndex < KERNEL_FH_K_ELEMS_PER_GRID) && (kGlobalIndex < numK);
         kIndex += 4, kGlobalIndex += 4) {
      float expArg = PIx2 *
        (c[kIndex].Kx * sX + c[kIndex].Ky * sY + c[kIndex].Kz * sZ);
      float cosArg = native_cos(expArg);
      float sinArg = native_sin(expArg);
      sOutR += c[kIndex].RhoPhiR * cosArg - c[kIndex].RhoPhiI * sinArg;
      sOutI += c[kIndex].RhoPhiI * cosArg + c[kIndex].RhoPhiR * sinArg;
      
      int kIndex1 = kIndex + 1;
      float expArg1 = PIx2 *
        (c[kIndex1].Kx * sX + c[kIndex1].Ky * sY + c[kIndex1].Kz * sZ);
      float cosArg1 = native_cos(expArg1);
      float sinArg1 = native_sin(expArg1);
      sOutR += c[kIndex1].RhoPhiR * cosArg1 - c[kIndex1].RhoPhiI * sinArg1;
      sOutI += c[kIndex1].RhoPhiI * cosArg1 + c[kIndex1].RhoPhiR * sinArg1;

      int kIndex2 = kIndex + 2;
      float expArg2 = PIx2 *
        (c[kIndex2].Kx * sX + c[kIndex2].Ky * sY + c[kIndex2].Kz * sZ);
      float cosArg2 = native_cos(expArg2);
      float sinArg2 = native_sin(expArg2);
      sOutR += c[kIndex2].RhoPhiR * cosArg2 - c[kIndex2].RhoPhiI * sinArg2;
      sOutI += c[kIndex2].RhoPhiI * cosArg2 + c[kIndex2].RhoPhiR * sinArg2;

      int kIndex3 = kIndex + 3;
      float expArg3 = PIx2 *
        (c[kIndex3].Kx * sX + c[kIndex3].Ky * sY + c[kIndex3].Kz * sZ);
      float cosArg3 = native_cos(expArg3);
      float sinArg3 = native_sin(expArg3);
      sOutR += c[kIndex3].RhoPhiR * cosArg3 - c[kIndex3].RhoPhiI * sinArg3;
      sOutI += c[kIndex3].RhoPhiI * cosArg3 + c[kIndex3].RhoPhiR * sinArg3;    
    }

    outR[xIndex] = sOutR;
    outI[xIndex] = sOutI;
  }
}


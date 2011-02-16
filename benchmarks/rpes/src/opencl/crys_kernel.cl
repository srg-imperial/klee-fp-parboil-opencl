/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/


void GetBases(int Work, int thid, int Shift, __private int *localthid, 
              __private int *a1, __private int *a2, __private int *a3, __private int *a4, int blockSizeLog)
{
  *a1         = (Work & (unsigned int)61440)   >> 12;
  *a2         = (Work & (unsigned int)3840 )   >> 8;
  *a3         = (Work & (unsigned int)240  )   >> 4;
  *a4         = (Work & (unsigned int)15   );
  *localthid = thid - (Shift << blockSizeLog);
}

void GetNs(int localthid, int a1, int a2, int a3, int a4, 
           __private int *n1, __private int *n2, __private int *n3, __private int *n4)
{
  int prod = mul24(a2, mul24(a3, a4));
  int res = localthid;
  *n1 = res / prod;
  res -= mul24(*n1, prod);

  prod = a3 * a4;
  *n2 = res / prod;
  res -= mul24(*n2, prod);

  prod = a4;
  *n3 = res / prod;
  res -= mul24(*n3, prod);

  *n4 = res;
}

void GetAtoms(int Work, __private int *atom1, __private int *atom2, __private int *atom3, __private int *atom4)
{
  *atom1 = (Work & (unsigned int)4278190080) >> 24;
  *atom2 = (Work & (unsigned int)16711680)   >> 16;
  *atom3 = (Work & (unsigned int)65280)      >> 8;
  *atom4 = (Work & (unsigned int)255);
}

void GetOffsets(int Work, __private int* off1, __private int *off2, __private int *off3, __private int *off4)
{
  *off1 = (Work & (unsigned int)4278190080) >> 24;
  *off2 = (Work & (unsigned int)16711680)   >> 16;
  *off3 = (Work & (unsigned int)65280)      >> 8;
  *off4 = (Work & (unsigned int)255);
}

float dist2(float4 Atom1, float4 Atom2)
{
  float dx = Atom1.x - Atom2.x;
  float dy = Atom1.y - Atom2.y;
  float dz = Atom1.z - Atom2.z;
  return dx * dx + dy * dy + dz * dz;
}

float product1D(float alpha_a, float coor_a, float alpha_b, 
                           float coor_b, float sum_ab)
{
  //return divide(alpha_a * coor_a + alpha_b * coor_b, sum_ab);
  return (alpha_a * coor_a + alpha_b * coor_b) / sum_ab;
}

float4 ReducePair(float4 Atom1, float4 Atom2, float2 Param1, 
                  float2 Param2, float sum_12)
{
  float4 Atomp;
  Atomp.x = product1D(Param1.x, Atom1.x, Param2.x, Atom2.x, sum_12);
  Atomp.y = product1D(Param1.x, Atom1.y, Param2.x, Atom2.y, sum_12);
  Atomp.z = product1D(Param1.x, Atom1.z, Param2.x, Atom2.z, sum_12);
  Atomp.w = 0;
  
  return Atomp;
}

float Root(float X)
{
  float rPIE4;
  float WW1 = 0.0f;
  float F1,E,Y,inv;
  
  rPIE4 = 1.273239545f;
  if (X < 3.0e-7f)
    {
      WW1 = 1.0f - 0.333333333f * X;
    } 
  else if (X < 1.0f) 
    {
      F1 = ((((((((-8.36313918003957E-08f*X+1.21222603512827E-06f )*X-
                  1.15662609053481E-05f )*X+9.25197374512647E-05f )*X-
                6.40994113129432E-04f )*X+3.78787044215009E-03f )*X-
              1.85185172458485E-02f )*X+7.14285713298222E-02f )*X-
            1.99999999997023E-01f )*X+3.33333333333318E-01f;
      WW1 = (X+X)*F1 + exp(-X);
    } 
  else if (X < 3.0f) 
    {
      Y = X-2.0f;
      F1 = ((((((((((-1.61702782425558E-10f*Y+1.96215250865776E-09f )*Y-
                    2.14234468198419E-08f )*Y+2.17216556336318E-07f )*Y-
                  1.98850171329371E-06f )*Y+1.62429321438911E-05f )*Y-
                1.16740298039895E-04f )*Y+7.24888732052332E-04f )*Y-
              3.79490003707156E-03f )*Y+1.61723488664661E-02f )*Y-
            5.29428148329736E-02f )*Y+1.15702180856167E-01f;
      WW1 = (X+X)*F1+exp(-X);
    } 
  else if (X < 5.0f)
    {
      Y = X-4.0f;
      F1 = ((((((((((-2.62453564772299E-11f*Y+3.24031041623823E-10f )*Y-
                    3.614965656163E-09f)*Y+3.760256799971E-08f)*Y-
                  3.553558319675E-07f)*Y+3.022556449731E-06f)*Y-
                2.290098979647E-05f)*Y+1.526537461148E-04f)*Y-
              8.81947375894379E-04f)*Y+4.33207949514611E-03f )*Y-
            1.75257821619926E-02f )*Y+5.28406320615584E-02f;
      WW1 = (X+X)*F1+exp(-X);
    } 
  else if (X < 10.0f) 
    {
      E = exp(-X);
      inv = 1 / X;
      WW1 = (((((( 4.6897511375022E-01f*inv-6.9955602298985E-01f)*inv +
                 5.3689283271887E-01f)*inv-3.2883030418398E-01f)*inv +
               2.4645596956002E-01f)*inv-4.9984072848436E-01f)*inv -
             3.1501078774085E-06f)*E + 1 / sqrt(rPIE4 * X);
    } 
  else if (X < 15.0f) 
    {
      E = exp(-X);
      inv = 1 / X;
      WW1 = (((-1.8784686463512E-01f*inv+2.2991849164985E-01f)*inv -
              4.9893752514047E-01f)*inv-2.1916512131607E-05f)*E \
        + 1 / sqrt(rPIE4 * X);
    } 
  else if (X < 33.0f) 
    {
      E = exp(-X);
      inv = 1 / X;
      WW1 = (( 1.9623264149430E-01f*inv-4.9695241464490E-01f)*inv -
             6.0156581186481E-05f)*E + 1 / sqrt(rPIE4 * X);
    } 
  else 
    {
      WW1 = 1 / sqrt(rPIE4 * X);
    }
  return WW1;
}


__kernel  void ComputeX(__global uint4* d_Work, __global float* d_Output, int Offset, __global float4* d_Coors,
              __global float2 *d_Sprms, __local float *Data)
{
  int blid = get_group_id(0);
  int thid = get_local_id(0);

  int blockSize = get_local_size(0);
  int blockSizeLog = (int)log2((float)blockSize);
  
  int myWorkIndex = ((Offset + blid) << blockSizeLog) + thid;
  uint4 myWork = d_Work[Offset + blid];
  
  int localthid, a1, a2, a3, a4;
  GetBases(myWork.x, myWorkIndex, myWork.w,
           &localthid, &a1, &a2, &a3, &a4, blockSizeLog);
  
  float Result = 0.0f;
  int Maxthid = mul24(a1, mul24(a2, mul24(a3, a4)));
  if(localthid < Maxthid)
    {
      int n1, n2, n3, n4;
      int off1, off2, off3, off4;
      int atom1, atom2, atom3, atom4;
      
      GetAtoms(myWork.y, &atom1, &atom2, &atom3, &atom4);
      GetNs(localthid, a1, a2, a3, a4, &n1, &n2, &n3, &n4);
      GetOffsets(myWork.z, &off1, &off2, &off3, &off4);
      n1 += off1;
      n2 += off2;
      n3 += off3;
      n4 += off4;
      
      float4 Atom1 = d_Coors[atom1];
      float4 Atom2 = d_Coors[atom2];
      float4 Atom3 = d_Coors[atom3];
      float4 Atom4 = d_Coors[atom4];
      float2 Param1 = d_Sprms[n1];
      float2 Param2 = d_Sprms[n2];
      float2 Param3 = d_Sprms[n3];
      float2 Param4 = d_Sprms[n4];


      float2 dummy; dummy.x = 0.123; dummy.y = 0.456;
      Param1 = Param2 = Param3 = Param4 = dummy;
      
      float R12 = dist2(Atom1, Atom2);
      float R34 = dist2(Atom3, Atom4);
      float sum12  = Param1.x + Param2.x;
      float sum34  = Param3.x + Param4.x;
      float prod12 = Param1.x * Param2.x;
      float prod34 = Param3.x * Param4.x;
      float preexp = native_divide (prod12, sum12) * R12 + 
                     prod34 / sum34 * R34;
      float preintegral = Param1.y * Param2.y * Param3.y * 
                          Param4.y * exp(- preexp) / 
                          (sum12 * sum34) * (1 / sqrt(sum12 + sum34));

      if(preintegral * preintegral > 1.0e-23f)
        {
          float4 Atomp = ReducePair(Atom1, Atom2, Param1, Param2, sum12);
          float4 Atomq = ReducePair(Atom3, Atom4, Param3, Param4, sum34);
          
          float rpq2 = dist2(Atomp, Atomq);
          float rho = native_divide (sum12 * sum34, (sum12 + sum34));
          float weight = Root(rpq2 * rho);
          Result = 34.98683666f * preintegral * weight;
        }
    }
  Data[thid] = Result;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(unsigned int s = get_local_size(0)>>1; s > 0;  s = s>>1)
    {
      if (thid < s)
        Data[thid] += Data[thid+s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  if(!thid)
    d_Output[Offset + blid] = Data[0];
}

__kernel  void DoReduction(__global float* d_ReductionSum, __global float* d_Output, 
                            __global uint2* d_FinalReduce, int Offset, __local float *Result)
{
  int firstElement;
  int offset;
  
  int blid = get_group_id(0);
  int thid = get_local_id(0);
  
  uint2 myWork = d_FinalReduce[blid + Offset];
  firstElement = myWork.x;
  offset = myWork.y;
  
  if(thid < offset)
    Result[thid] = d_Output[firstElement + thid];
  else
    Result[thid] = 0.0f;
  barrier(CLK_LOCAL_MEM_FENCE);

  for(unsigned int s = get_local_size(0)>>1; s > 0;  s = s>>1)
    {
      if (thid < s)
        Result[thid] += Result[thid+s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  if(!thid)
    d_ReductionSum[blid + Offset] = Result[0];
}



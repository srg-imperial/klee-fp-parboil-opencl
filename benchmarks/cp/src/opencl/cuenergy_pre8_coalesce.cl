/***************************************************************************
 *cr
 *cr            (C) Copyright 2007 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ***************************************************************************/

// Max constant buffer size is 64KB, minus whatever
// the CUDA runtime and compiler are using that we don't know about.
// At 16 bytes for atom, for this program 4070 atoms is about the max
// we can store in the constant buffer.
//__constant__ float4 atominfo[MAXATOMS];

// This kernel calculates coulombic potential at each grid point and
// stores the results in the output array.

#define UNROLLX 8

__kernel void cenergy(int numatoms, float gridspacing, __global float * energygrid, __constant float4 *atominfo) {
  unsigned int xindex  = get_group_id(0)*get_local_size(0) * UNROLLX + get_local_id(0);
  unsigned int yindex  = get_global_id(1);
  unsigned int outaddr = get_global_size(0) * UNROLLX * yindex + xindex;
  int blockSizeX = get_local_size(0);

  float coory = yindex * gridspacing;
  float8 coorx = gridspacing *
   (float8)(xindex,              xindex+  blockSizeX, xindex+2*blockSizeX, xindex+3*blockSizeX,
            xindex+4*blockSizeX, xindex+5*blockSizeX, xindex+6*blockSizeX, xindex+7*blockSizeX);

  float energyvalx1=0.0f;
  float energyvalx2=0.0f;
  float energyvalx3=0.0f;
  float energyvalx4=0.0f;
  float energyvalx5=0.0f;
  float energyvalx6=0.0f;
  float energyvalx7=0.0f;
  float energyvalx8=0.0f;

  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dyz2 = (dy * dy) + atominfo[atomid].z;

    float8 dx = coorx - atominfo[atomid].x;

    energyvalx1 += atominfo[atomid].w / native_sqrt(dx[0]*dx[0] + dyz2);
    energyvalx2 += atominfo[atomid].w / native_sqrt(dx[1]*dx[1] + dyz2);
    energyvalx3 += atominfo[atomid].w / native_sqrt(dx[2]*dx[2] + dyz2);
    energyvalx4 += atominfo[atomid].w / native_sqrt(dx[3]*dx[3] + dyz2);
    energyvalx5 += atominfo[atomid].w / native_sqrt(dx[4]*dx[4] + dyz2);
    energyvalx6 += atominfo[atomid].w / native_sqrt(dx[5]*dx[5] + dyz2);
    energyvalx7 += atominfo[atomid].w / native_sqrt(dx[6]*dx[6] + dyz2);
    energyvalx8 += atominfo[atomid].w / native_sqrt(dx[7]*dx[7] + dyz2);
  }

  energygrid[outaddr]   += energyvalx1;
  energygrid[outaddr+1*blockSizeX] += energyvalx2;
  energygrid[outaddr+2*blockSizeX] += energyvalx3;
  energygrid[outaddr+3*blockSizeX] += energyvalx4;
  energygrid[outaddr+4*blockSizeX] += energyvalx5;
  energygrid[outaddr+5*blockSizeX] += energyvalx6;
  energygrid[outaddr+6*blockSizeX] += energyvalx7;
  energygrid[outaddr+7*blockSizeX] += energyvalx8;
}


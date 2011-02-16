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

  float coory = gridspacing * yindex;
  float coorx = gridspacing * xindex;

  float energyvalx1=0.0f;
  float energyvalx2=0.0f;
  float energyvalx3=0.0f;
  float energyvalx4=0.0f;
  float energyvalx5=0.0f;
  float energyvalx6=0.0f;
  float energyvalx7=0.0f;
  float energyvalx8=0.0f;

  int blockSizeX = get_local_size(0);
  float gridspacing_u = gridspacing * blockSizeX;

  int atomid;
  for (atomid=0; atomid<numatoms; atomid++) {
    float dy = coory - atominfo[atomid].y;
    float dyz2 = (dy * dy) + atominfo[atomid].z;

    float dx1 = coorx - atominfo[atomid].x;
    float dx2 = dx1 + gridspacing_u;
    float dx3 = dx2 + gridspacing_u;
    float dx4 = dx3 + gridspacing_u;
    float dx5 = dx4 + gridspacing_u;
    float dx6 = dx5 + gridspacing_u;
    float dx7 = dx6 + gridspacing_u;
    float dx8 = dx7 + gridspacing_u;

    energyvalx1 += atominfo[atomid].w * (1.0f / native_sqrt(dx1*dx1 + dyz2));
    energyvalx2 += atominfo[atomid].w * (1.0f / native_sqrt(dx2*dx2 + dyz2));
    energyvalx3 += atominfo[atomid].w * (1.0f / native_sqrt(dx3*dx3 + dyz2));
    energyvalx4 += atominfo[atomid].w * (1.0f / native_sqrt(dx4*dx4 + dyz2));
    energyvalx5 += atominfo[atomid].w * (1.0f / native_sqrt(dx5*dx5 + dyz2));
    energyvalx6 += atominfo[atomid].w * (1.0f / native_sqrt(dx6*dx6 + dyz2));
    energyvalx7 += atominfo[atomid].w * (1.0f / native_sqrt(dx7*dx7 + dyz2));
    energyvalx8 += atominfo[atomid].w * (1.0f / native_sqrt(dx8*dx8 + dyz2));
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


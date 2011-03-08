struct kValues {
  float Kx;
  float Ky;
  float Kz;
  float RealRhoPhi;
  float ImagRhoPhi;
};

void 
ComputeRhoPhi(int numK,
              float* __restrict__ phiR, float* __restrict__ phiI,
              float* __restrict__ dR, float* __restrict__ dI,
              float* __restrict__ realRhoPhi, float* __restrict__ imagRhoPhi);

void
ComputeFH(int numK, int numX, 
	  kValues * __restrict__ kVals,
          float * __restrict__ x,
	  float * __restrict__ y,
	  float * __restrict__ z,
          float * __restrict__ outR, float * __restrict__ outI);

void createDataStructs(int numK, int numX, 
                       float*& realRhoPhi, float*& imagRhoPhi, 
                       float*& outR, float*& outI);

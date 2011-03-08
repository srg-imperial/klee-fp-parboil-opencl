#include <stddef.h>

#ifdef __cplusplus
extern "C"
#endif
int computeMRIFHD_GPU(size_t rhophiWorkSize, size_t fhWorkSize, int numK, int numX,
  float *kx, float *ky, float *kz,
  float *x, float *y, float *z,
  float *phiR, float *phiI,
  float *dR, float *dI,
  float **outR, float **outI);

#include <stddef.h>

#ifdef __cplusplus
extern "C"
#endif
int computeMRIQ_GPU(size_t phiWorkSize, size_t qWorkSize, int numK, int numX, 
                    float *kx, float *ky, float *kz,
                    float *x, float *y, float *z,
                    float *phiR, float *phiI,
                    float **Qr, float **Qi);

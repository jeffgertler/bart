#ifndef _KERNELS_H_
#define _KERNELS_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC double gp_isotropic_kernel (double x1, double x2, int npars,
                                    double *pars);
EXTERNC void gp_isotropic_kernel_grad (double x1, double x2, int npars,
                                       double *pars, double *dkdt);

#undef EXTERNC

#endif
// /_KERNELS_H_

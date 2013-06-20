#ifndef _GEORGE_H_
#define _GEORGE_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC double gp_lnlikelihood (int nsamples, double *x, double *y,
                                double *yerr, double amp, double var);
EXTERNC int gp_predict (int nsamples, double *x, double *y, double *yerr,
                        double amp, double var, int ntest, double *xtest,
                        double *ytest);
EXTERNC double gp_isotropic_kernel (double x1, double x2, double amp,
                                       double var);

#undef EXTERNC

#endif
// /_GEORGE_H_

#ifndef _TURNSTILE_H_
#define _TURNSTILE_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC void turnstile (int nsets, int *ndata, double **time, double **flux,
                        double **ferr, double amp, double var,
                        double min_period, double max_period, int nperiods,
                        double min_depth, double max_depth, double d_depth,
                        double *depths)

#undef EXTERNC

#endif
// /_TURNSTILE_H_

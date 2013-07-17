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
                        int **nepochs, double **periods, double ***epochs,
                        double ***depths, double ***dvar);

#undef EXTERNC

#endif
// /_TURNSTILE_H_

#ifndef _BART_H_
#define _BART_H_

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC int lightcurve (int n, double *t, double *flux,
                        int nbin, double texp,
                        double fstar, double mstar, double rstar, double iobs,
                        int np, double *mass, double *r, double *a, double *t0,
                        double *e, double *pomega, double *ix, double *iy,
                        int nld, double *rld, double *ild);

EXTERNC void fast_lightcurve (int n, double *time, double *flux,
                              double rp, double period, double t0p,
                              int nld, double *rld, double *ild);

#undef EXTERNC

#endif
/* _BART_H_ */

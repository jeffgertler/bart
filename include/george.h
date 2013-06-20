#ifndef _GEORGE_H_
#define _GEORGE_H_

#include <Eigen/Dense>

double gp_lnlikelihood (int nsamples, double *x, double *y, double *yerr,
                        int npars, double *kpars,
                        double (*kernel) (double, double, int, double*));

int gp_predict (int nsamples, double *x, double *y, double *yerr,
                int npars, double *kpars,
                double (*kernel) (double, double, int, double*),
                int ntest, double *xtest, double *ytest);

double gp_isotropic_kernel (double x1, double x2, int npars, double *pars);

#endif
// /_GEORGE_H_

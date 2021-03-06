#ifndef _GEORGE_H_
#define _GEORGE_H_

#ifdef __cplusplus
#include <Eigen/Dense>

#define EXTERNC extern "C"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::LDLT;

class George
{

    private:

        int nsamples_;
        double *x_;

        int npars_;
        double *pars_;
        double (*kernel_) (double x1, double x2, int npars, double *pars);
        void (*dkernel_) (double x1, double x2, int npars, double *pars,
                          double *dkdt);

        bool computed_;
        LDLT<MatrixXd> L_;

    public:

        George (int npars, double *pars,
                double (*kernel) (double, double, int, double*),
                void (*dkernel) (double, double, int, double*, double*)) {
            computed_ = false;
            npars_ = npars;
            pars_ = pars;
            kernel_ = kernel;
            dkernel_ = dkernel;
            nsamples_ = 0;
        };

        int compute (int nsamples, double *x, double *yerr);
        double lnlikelihood (int nsamples, double *y);
        int gradlnlikelihood (int nsamples, double *y, double *lnlike,
                              VectorXd *gradlnlike);
        int predict (int nsamples, double *y, int ntest, double *x,
                     double *mu, double *cov);

};

#else
#define EXTERNC
#endif

EXTERNC double gp_lnlikelihood (int nsamples, double *x, double *y,
                                double *yerr, double amp, double var);
EXTERNC int gp_predict (int nsamples, double *x, double *y, double *yerr,
                        double amp, double var, int ntest, double *xtest,
                        double *mean, double *cov);
EXTERNC int gp_gradlnlikelihood (int nsamples, double *x, double *y,
                                 double *yerr, double amp, double var,
                                 double *lnlike, double **gradlnlike);

#undef EXTERNC

#endif
// /_GEORGE_H_

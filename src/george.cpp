#include "george.h"

#define TWOLNPI 1.8378770664093453

using namespace Eigen;

int George::compute (int nsamples, double *x, double *yerr)
{
    int i, j;
    double value;
    LDLT<MatrixXd> L;
    MatrixXd Kxx(nsamples, nsamples);

    nsamples_ = nsamples;
    x_ = x;

    for (i = 0; i < nsamples; ++i) {
        for (j = i + 1; j < nsamples; ++j) {
            value = kernel_(x[i], x[j], npars_, pars_);
            Kxx(i, j) = value;
            Kxx(j, i) = value;
        }
        Kxx(i, i) = kernel_(x[i], x[i], npars_, pars_) + yerr[i] * yerr[i];
    }

    L_ = LDLT<MatrixXd>(Kxx);
    if (L.info() != Success) return -1;

    computed_ = true;
    return 0;
}

double George::lnlikelihood (int nsamples, double *y)
{
    double logdet;
    VectorXd alpha, yvec = Map<VectorXd>(y, nsamples);

    if (!computed_ || nsamples != nsamples_)
        return -INFINITY;

    alpha = L_.solve(yvec);
    if (L_.info() != Success)
        return -INFINITY;

    logdet = log(L_.vectorD().array()).sum();
    return -0.5 * (yvec.transpose() * alpha + logdet + nsamples * TWOLNPI);
}

int George::predict (int nsamples, double *y, int ntest, double *x,
                     double *mu, double *cov)
{
    int i, j;
    double value;
    MatrixXd Kxs(nsamples_, ntest), Kss(ntest, ntest);
    VectorXd alpha, ytest, yvec = Map<VectorXd>(y, nsamples);

    if (!computed_ || nsamples != nsamples_)
        return -1;

    // Build the kernel matrices.
    for (i = 0; i < nsamples; ++i)
        for (j = 0; j < ntest; ++j)
            Kxs(i, j) = kernel_(x_[i], x[j], npars_, pars_);
    for (i = 0; i < ntest; ++i) {
        Kss(i, i) = kernel_(x[i], x[i], npars_, pars_);
        for (j = i + 1; j < ntest; ++j) {
            value = kernel_(x[i], x[j], npars_, pars_);
            Kss(i, j) = value;
            Kss(j, i) = value;
        }
    }

    alpha = L_.solve(yvec);
    if (L_.info() != Success)
        return -2;

    // Compute and copy the predictive mean vector.
    ytest = Kxs.transpose() * alpha;
    for (i = 0; i < ntest; ++i) mu[i] = ytest[i];

    // Compute the predictive covariance matrix.
    Kss -= Kxs.transpose() * L_.solve(Kxs);
    if (L_.info() != Success)
        return -3;
    for (i = 0; i < ntest; ++i)
        for (j = 0; j < ntest; ++j)
            cov[i * ntest + j] = Kss(i, j);

    return 0;
}

// C interface functions.

double gp_lnlikelihood (int nsamples, double *x, double *y, double *yerr,
                        double amp, double var)
{
    int info;
    double pars[2] = {amp, var};
    George gp(2, pars, &gp_isotropic_kernel);
    info = gp.compute(nsamples, x, yerr);
    if (info != 0)
        return -INFINITY;
    return gp.lnlikelihood(nsamples, y);
}

int gp_predict (int nsamples, double *x, double *y, double *yerr, double amp,
                double var, int ntest, double *xtest, double *mean,
                double *cov)
{
    int info;
    double pars[2] = {amp, var};
    George gp(2, pars, &gp_isotropic_kernel);
    info = gp.compute(nsamples, x, yerr);
    if (info != 0)
        return info;
    return gp.predict(nsamples, y, ntest, xtest, mean, cov);
}

double gp_isotropic_kernel (double x1, double x2, int npars, double *pars)
{
    int i;
    double d = x1 - x2, chi2 = -0.5 * d * d, value = 0.0;
    for (i = 0; i + 1 < npars; i += 2)
        value += pars[i] * exp(chi2 / pars[i + 1]);
    return value;
}

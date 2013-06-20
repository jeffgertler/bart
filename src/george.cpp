#include "george.h"

using namespace Eigen;

double gp_lnlikelihood (int nsamples, double *x, double *y, double *yerr,
                        int npars, double *kpars,
                        double (*kernel) (double, double, int, double*))
{
    int i, j;
    double logdet;
    MatrixXd Kxx(nsamples, nsamples);
    LDLT<MatrixXd> L;
    VectorXd alpha, yvec = Map<VectorXd>(y, nsamples);

    // Build the kernel matrix.
    for (i = 0; i < nsamples; ++i) {
        for (j = 0; j < nsamples; ++j)
            Kxx(i, j) = kernel(x[i], x[j], npars, kpars);
        Kxx(i, i) += yerr[i] * yerr[i];
    }

    // Compute the decomposition of K(X, X)
    L = LDLT<MatrixXd>(Kxx);
    if (L.info() != Success) {
        printf("Decomposition failed.\n");
        return -INFINITY;
    }

    // Solve the system.
    alpha = L.solve(yvec);
    if (L.info() != Success) {
        printf("Solve failed.\n");
        return -INFINITY;
    }

    // Compute the likelihood value.
    logdet = log(L.vectorD().array()).sum();
    return -0.5 * (yvec.transpose() * alpha + logdet
           + nsamples * log(2 * M_PI));
}

int gp_predict (int nsamples, double *x, double *y, double *yerr,
                int npars, double *kpars,
                double (*kernel) (double, double, int, double*),
                int ntest, double *xtest, double *ytest)
{
    int i, j;
    double logdet;
    MatrixXd Kxx(nsamples, nsamples), Kstar(nsamples, ntest);
    LDLT<MatrixXd> L;
    VectorXd alpha, mean, yvec = Map<VectorXd>(y, nsamples);

    // Build the kernel matrix.
    for (i = 0; i < nsamples; ++i) {
        for (j = 0; j < nsamples; ++j)
            Kxx(i, j) = kernel(x[i], x[j], npars, kpars);
        Kxx(i, i) += yerr[i] * yerr[i];
        for (j = 0; j < ntest; ++j)
            Kxx(i, j) = kernel(x[i], xtest[j], npars, kpars);
    }

    // Compute the decomposition of K(X, X)
    L = LDLT<MatrixXd>(Kxx);
    if (L.info() != Success) {
        printf("Decomposition failed.\n");
        return -1;
    }

    // Solve the system.
    alpha = L.solve(yvec);
    if (L.info() != Success) {
        printf("Solve failed.\n");
        return -2;
    }

    mean = Kstar.transpose() * alpha;
    for (i = 0; i < ntest; ++i)
        ytest[i] = mean[i];

    return 0;
}

double gp_isotropic_kernel(double x1, double x2, int npars, double *pars)
{
    double d = x1 - x2;
    return pars[0] * exp(-0.5 * d * d * pars[1]);
}

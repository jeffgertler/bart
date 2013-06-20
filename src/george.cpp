#include <Eigen/Dense>
#include "george.h"

using namespace Eigen;

double gp_lnlikelihood (int nsamples, double *x, double *y,
                                   double *yerr, double amp, double var)
{
    int i, j;
    double logdet;
    MatrixXd Kxx(nsamples, nsamples);
    LDLT<MatrixXd> L;
    VectorXd alpha, yvec = Map<VectorXd>(y, nsamples);

    // Build the kernel matrix.
    for (i = 0; i < nsamples; ++i) {
        for (j = 0; j < nsamples; ++j)
            Kxx(i, j) = gp_isotropic_kernel(x[i], x[j], amp, var);
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
                           double amp, double var, int ntest, double *xtest,
                           double *ytest)
{
    int i, j;
    MatrixXd Kxx(nsamples, nsamples), Kstar(nsamples, ntest);
    LDLT<MatrixXd> L;
    VectorXd alpha, mean, yvec = Map<VectorXd>(y, nsamples);
    printf("supfoo\n");

    // Build the kernel matrix.
    for (i = 0; i < nsamples; ++i) {
        for (j = 0; j < nsamples; ++j)
            Kxx(i, j) = gp_isotropic_kernel(x[i], x[j], amp, var);
        Kxx(i, i) += yerr[i] * yerr[i];
        for (j = 0; j < ntest; ++j)
            Kstar(i, j) = gp_isotropic_kernel(x[i], xtest[j], amp, var);
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
    for (i = 0; i < ntest; ++i) {
        ytest[i] = mean[i];
    }

    return 0;
}

double gp_isotropic_kernel (double x1, double x2, double amp,
                                       double var)
{
    double d = x1 - x2;
    return amp * exp(-0.5 * d * d / var);
}

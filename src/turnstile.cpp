#include "turnstile.h"
#include "kernels.h"
#include "kepler.h"
#include "bart.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::LDLT;
using Eigen::Success;

int max_like_depth (VectorXd time, VectorXd flux, VectorXd ferr,
                    VectorXd model, int npars, double *pars,
                    double (*kernel) (double, double, int, double*),
                    double *numerator, double *denominator)
{
    int i, j, nsamples = time.size();
    double value;
    LDLT<MatrixXd> L;
    MatrixXd Kxx(nsamples, nsamples);

    // Build the covariance matrix.
    for (i = 0; i < nsamples; ++i) {
        for (j = i + 1; j < nsamples; ++j) {
            value = kernel(time[i], time[j], npars, pars);
            Kxx(i, j) = value;
            Kxx(j, i) = value;
        }
        Kxx(i, i) = kernel(time[i], time[i], npars, pars) + ferr[i] * ferr[i];
    }

    // Factorize.
    L = LDLT<MatrixXd>(Kxx);
    if (L.info() != Success) return -1;

    // Compute the maximum likelihood depth (up to a factor of the variance).
    *numerator = model.dot(L.solve(flux));
    if (L.info() != Success) return -2;

    // Compute the variance on the estimated depth.
    *denominator = model.dot(L.solve(model));
    if (L.info() != Success) return -3;

    return 0;
}

void turnstile (int nsets, int *ndata, double **time, double **flux,
                double **ferr, double amp, double var,
                double min_period, double max_period, int nperiods,
                int **nepochs, double **periods, double ***epochs,
                double ***depths, double ***dvar)
{
    int i, j, l, np, ne, te, npts, info;
    double period, epoch, duration, tfold, ntmp, dtmp, numerator,
           denominator, pars[2] = {amp, var};

    // Allocate some memory.
    *nepochs = (int*)malloc(nperiods * sizeof(int));
    *periods = (double*)malloc(nperiods * sizeof(double));
    *epochs = (double**)malloc(nperiods * sizeof(double*));
    *depths = (double**)malloc(nperiods * sizeof(double*));
    *dvar = (double**)malloc(nperiods * sizeof(double*));

    for (np = 0; np < nperiods; ++np) {
        period = min_period + np * (max_period - min_period) / (nperiods - 1);
        (*periods)[np] = period;
        printf("Period %d: %e\n", np, period);

        // MAGIC: compute the shit out of the duration. Don't ask.
        // duration = 0.5 * exp(0.44 * log(period) - 2.97);
        duration = pow(period / (2 * M_PI * G_GRAV), 1. / 3.);

        // Figure out the number of epochs.
        te = (*nepochs)[np] = period / (0.1 * duration);

        // Allocate the inner data structures for the depth and epoch arrays.
        (*epochs)[np] = (double*)malloc(te * sizeof(double));
        (*depths)[np] = (double*)malloc(te * sizeof(double));
        (*dvar)[np] = (double*)malloc(te * sizeof(double));

        // Loop over epochs.
        for (ne = 0; ne < te; ++ne) {
            (*epochs)[np][ne] = epoch = ne * period / (te - 1);

            numerator = 0.0;
            denominator = 0.0;
            for (i = 0; i < nsets; ++i) {
                VectorXd t(ndata[i]), f(ndata[i]), fe(ndata[i]);

                // Figure out which data points are relevant for this fit.
                l = ndata[i];
                npts = 0;
                for (j = 0; j < l; ++j) {
                    tfold = fabs(fmod(time[i][j] - epoch + 0.5 * period, period)
                                 - 0.5 * period);
                    if (tfold < 3 * duration) {
                        t(npts) = time[i][j];
                        f(npts) = flux[i][j] - 1.0;
                        fe(npts) = ferr[i][j];
                        npts++;
                    }
                }

                if (npts > 0) {
                    // Resize the vectors.
                    t.conservativeResize(npts);
                    f.conservativeResize(npts);
                    fe.conservativeResize(npts);

                    VectorXd model(npts);
                    double rld[1] = {1.0}, ild[1] = {1.0};
                    fast_lightcurve(npts, &(t[0]), &(model[0]), 0.05, period,
                                    epoch, 1, rld, ild);
                    model = model.array() / model.maxCoeff() - 1.0;

                    // Solve for the depth.
                    info = max_like_depth (t, f, fe, model, 2, pars,
                                           &gp_isotropic_kernel, &ntmp, &dtmp);
                    if (info == 0) {
                        numerator += ntmp;
                        denominator += dtmp;
                    }
                }
            }

            if (denominator > 0.0) {
                (*depths)[np][ne] = numerator / denominator;
                (*dvar)[np][ne] = 1.0 / denominator;
            } else {
                (*depths)[np][ne] = 0.0;
            }
        }
    }
}

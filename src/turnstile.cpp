#include "turnstile.h"
#include <Eigen/Dense>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::LDLT;

double max_like_depth (VectorXd time, VectorXd flux, VectorXd ferr,
                       VectorXd model, int npars, double *pars,
                       double (*kernel) (double, double, int, double*))
{
    int i, j, nsamples = time.size();
    double value;
    LDLT<MatrixXd> L;
    MatrixXd Kxx(nsamples, nsamples);

    for (i = 0; i < nsamples; ++i) {
        for (j = i + 1; j < nsamples; ++j) {
            value = kernel(time[i], time[j], npars, pars);
            Kxx(i, j) = value;
            Kxx(j, i) = value;
        }
        Kxx(i, i) = kernel(time[i], time[i], npars, pars) + ferr[i] * ferr[i];
    }

    L = LDLT<MatrixXd>(Kxx);
    if (L.info() != Success) return -1;
}

void turnstile (int nsets, int *ndata, double **time, double **flux,
                double **ferr, double amp, double var,
                double min_period, double max_period, int nperiods,
                double min_depth, double max_depth, int ndepths,
                double *periods, double *depths)
{
    int i, j, l, np, nvalid;
    double period, epoch, duration, depth, bestdepth, ldepth, mdepth, weight,
           lnlike, t, tfold, ll;
    vector<LightCurve*> lcs;

    // Initialize the working datasets.
    for (i = 0; i < nsets; ++i)
        lcs.push_back(new LightCurve(amp, var));

    for (np = 0; np < nperiods; ++np) {
        period = min_period + np * (max_period - min_period) / (nperiods - 1);
        periods[np] = period;
        bestdepth = 0.0;

        // MAGIC: compute the shit out of the duration. Don't ask.
        duration = 0.5 * exp(0.44 * log(period) - 2.97);

        // Loop over epochs.
        for (epoch = 0; epoch < period; epoch += 0.5 * duration) {
            for (i = 0; i < nsets; ++i) {
                // Reset the dataset wrapper.
                lcs[i]->reset();

                // Figure out which data points are relevant for this fit.
                l = ndata[i];
                for (j = 0; j < l; ++j) {
                    t = time[i][j];
                    tfold = fabs(fmod(t - epoch + 0.5 * period, period)
                                 - 0.5 * period);
                    if (tfold < 3 * duration) {
                        lcs[i]->push_back(t, flux[i][j], ferr[i][j],
                                          tfold < duration);
                    }
                }

                // Compute/factorize the GP.
                lcs[i]->compute();
            }

            // Loop over depths and accumulate the marginal depth.
            mdepth = -INFINITY;
            weight = -INFINITY;
            for (j = 0; j < ndepths; ++j) {
                depth = min_depth + j * (max_depth - min_depth) / (ndepths - 1);
                if (depth > 0.0) {
                    lnlike = 0.0;
                    nvalid = 0;
                    for (i = 0; i < nsets; ++i) {
                        if (lcs[i]->is_valid()) {
                            ll = lcs[i]->update_depth(depth);
                            if (finite(ll)) {
                                lnlike += ll;
                                nvalid ++;
                            }
                        }
                    }

                    if (nvalid > 0) {
                        // Update the marginal accumulation and the weight.
                        if (finite(weight)) {
                            ldepth = log(depth);
                            mdepth = logsumexp(mdepth, ldepth + lnlike);
                            weight = logsumexp(weight, lnlike);
                        } else {
                            mdepth = log(depth) + lnlike;
                            weight = lnlike;
                        }
                    }
                }
            }

            // Exponentiate the result.
            printf("diff: %e - %e = %e\n", mdepth, weight, mdepth - weight);
            mdepth = exp(mdepth - weight);
            if (mdepth > bestdepth) {
                bestdepth = mdepth;
            }
        }

        // Add the best depth to the result vector.
        depths[np] = bestdepth;
        printf("%e %e\n", period, bestdepth);
    }

    for (i = 0; i < nsets; ++i)
        delete lcs[i];
}

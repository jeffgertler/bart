#include "turnstile.h"
#include "george.h"
#include <vector>

using std::vector;

class LightCurve
{

    private:

        double prev_depth_;
        bool valid_;
        vector<double> time_, flux_, ferr_;
        vector<int> in_transit_;
        George *gp_;

    public:

        LightCurve (double amp, double var);
        ~LightCurve ();
        void push_back (double time, double flux, double ferr,
                        bool in_transit);
        void reset ();
        int compute ();
        double update_depth (double depth);
        bool is_valid () { return valid_; };

};

LightCurve::LightCurve (double amp, double var)
{
    double pars[2] = {amp, var};
    gp_ = new George(2, pars, &gp_isotropic_kernel);
    prev_depth_ = 0.0;
    valid_ = false;
}

LightCurve::~LightCurve ()
{
    delete gp_;
}

void LightCurve::push_back (double time, double flux, double ferr,
                            bool in_transit)
{
    time_.push_back(time);
    flux_.push_back(flux - 1);
    ferr_.push_back(ferr);
    if (in_transit) in_transit_.push_back(time_.size() - 1);
}

void LightCurve::reset ()
{
    time_.clear();
    flux_.clear();
    ferr_.clear();
    prev_depth_ = 0.0;
    valid_ = false;
}

int LightCurve::compute ()
{
    int size = (int)(time_.size()), info;
    if (size > 0) info = gp_->compute(size, &(time_[0]), &(ferr_[0]));
    else info = -10;
    if (info == 0) valid_ = true;
    else valid_ = false;
    return info;
}

double LightCurve::update_depth (double depth)
{
    int i, ind, l = in_transit_.size();
    double factor = (1 - prev_depth_) / (1 - depth);

    if (!valid_) return -INFINITY;

    for (i = 0; i < l; ++i) {
        ind = in_transit_[i];
        flux_[ind] = (flux_[ind] + 1) * factor - 1;
    }
    prev_depth_ = depth;

    return gp_->lnlikelihood((int)(time_.size()), &(flux_[0]));
}

double logsumexp (double a, double b)
{
    double A = a;
    if (b > a) A = b;
    return A + log(exp(a - A) + exp(b - A));
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

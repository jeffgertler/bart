#include "turnstile.h"
#include "george.h"
#include <vector>

using std::vector;

class LightCurve
{

    private:

        double prev_depth_;
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

};

LightCurve::LightCurve (double amp, double var)
{
    double pars[2] = {amp, var};
    gp_ = new George(2, pars, &gp_isotropic_kernel);
    prev_depth_ = 0.0;
}

LightCurve::~LightCurve ()
{
    delete gp_;
}

void LightCurve::push_back (double time, double flux, double ferr,
                            bool in_transit)
{
    time_.push_back(time);
    flux_.push_back(time);
    ferr_.push_back(time);
    if (in_transit) in_transit_.push_back(time_.size() - 1);
}

void LightCurve::reset ()
{
    time_.clear();
    flux_.clear();
    ferr_.clear();
    prev_depth_ = 0.0;
}

int LightCurve::compute ()
{
    return gp_->compute((int)(time_.size()), &(time_[0]), &(ferr_[0]));
}

double LightCurve::update_depth (double depth)
{
    int i, l = in_transit_.size();
    double factor = (1 - prev_depth_) / (1 - depth);
    for (i = 0; i < l; ++i) flux_[i] *= factor;
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
                double min_depth, double max_depth, double d_depth,
                double *depths)
{
    int i, j, l, np;
    double period, epoch, duration, depth, maxdepth, ldepth, mdepth, weight,
           lnlike, t, tfold;
    vector<LightCurve*> lcs;

    // Initialize the working datasets.
    for (i = 0; i < nsets; ++i)
        lcs.push_back(new LightCurve(amp, var));

    for (np = 0; np < nperiods; ++np) {
        period = min_period + (max_period - min_period) / (nperiods - 1);
        maxdepth = 0.0;

        // MAGIC: compute the shit out of the duration. Don't ask.
        duration = 0.5 * exp(0.44 * log(period) - 2.97);

        // Loop over epochs.
        for (epoch = 0; epoch < period; epoch += 0.25 * duration) {
            for (i = 0; i < nsets; ++i) {
                // Reset the dataset wrapper.
                lcs[i]->reset();

                // Figure out which data points are relevant for this fit.
                l = ndata[i];
                for (j = 0; j < l; ++j) {
                    t = time[i][j];
                    tfold = fabs(fmod(t - epoch + 0.5 * period, period)
                                 - 0.5 * period);
                    if (tfold < 3 * duration)
                        lcs[i]->push_back(t, flux[i][j], ferr[i][j],
                                          tfold < duration);
                }
            }

            // Loop over depths and accumulate the marginal depth.
            mdepth = 0.0;
            weight = 0.0;
            for (depth = min_depth; depth <= max_depth; depth += d_depth) {
                if (depth > 0.0) {
                    lnlike = 0.0;
                    for (i = 0; i < nsets; ++i)
                        lnlike += lcs[i]->update_depth(depth);

                    // Update the marginal accumulation and the weight.
                    ldepth = log(depth);
                    mdepth = logsumexp(mdepth, ldepth + lnlike);
                    weight = logsumexp(weight, lnlike);
                }
            }

            // Exponentiate the result.
            mdepth = exp(mdepth - weight);
            if (mdepth > maxdepth) {
                maxdepth = mdepth;
            }
        }

        // Add the best depth to the result vector.
        depths[np] = maxdepth;
    }

    for (i = 0; i < nsets; ++i)
        delete lcs[i];
}

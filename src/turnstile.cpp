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
        void update_depth (double depth);

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

void LightCurve::update_depth (double depth)
{
    int i, l = in_transit_.size();
    double factor = (1 - prev_depth_) / (1 - depth);
    for (i = 0; i < l; ++i) flux_[i] *= factor;
    prev_depth_ = depth;
}

double turnstile_marginalized_depth (int nsets, Dataset **datasets,
                                     double period, double epoch,
                                     double duration,
                                     double rmin, double rmax, double dr,
                                     double amp, double var)
{
    int info;
    double r, pars[2] = {amp, var}, *ftmp = malloc;

    // Compute the initial GP stuff.
    info = gp.compute(n, t, ferr);
    if (info != 0)
        return -INFINITY;

    // Loop over depth.
    for (r = rmin; r <= rmax; r += dr) {

    }
}

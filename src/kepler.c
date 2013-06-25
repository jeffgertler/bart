#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kepler.h"

double wt2psi(double wt, double e, int *info)
{
    int i;
    double wt0, psi0, psi = 0.0;

    // Check for un-physical parameters.
    if (e < 0 || e >= 1) {
        *info = 2;
        return 0.0;
    }

    *info = 0;
    wt0 = fmod(wt, 2 * M_PI);
    psi0 = wt0;
    for (i = 0; i < 500; ++i) {
        double spsi0 = sin(psi0),
               f = psi0 - e * spsi0 - wt0,
               fp = 1.0 - e * cos(psi0),
               fpp = e * spsi0;

        // Take a second order step.
        psi = psi0 - 2.0 * f * fp / (2.0 * fp * fp - f * fpp);

        // Accept?
        if (fabs(psi - psi0) <= 1.48e-10) return psi;

        // Save as the previous step.
        psi0 = psi;
    }

    *info = 1;
    return psi;
}

int solve_orbit(int n, double *t, double *pos,
                double mstar, double mplanet, double e, double a, double t0,
                double pomega, double ix, double iy)
{
    int i, info = 0;
    double period = 2 * M_PI * sqrt(a * a * a / G_GRAV / (mstar + mplanet)),
           psi0 = 2 * atan2(tan(0.5 * pomega), sqrt((1 + e) / (1 - e))),
           t1 = t0 -  0.5 * period * (psi0 - e * sin(psi0)) / M_PI;

    for (i = 0; i < n; ++i) {
        double manom = 2 * M_PI * (t[i] - t1) / period,
               psi = wt2psi(manom, e, &info);

        // Did solve fail?
        if (info != 0) return info;

        // Rotate into the observed frame.
        double cpsi = cos(psi), spsi = sin(psi), d = 1.0 - e * cpsi,
               cth = (cpsi - e) / d, sth = sqrt(1 - cth * cth),
               r = a * d,
               x = r * cth,
               y = r * spsi * ((sth > 0) - (sth < 0)),

               // Rotate by pomega.
               xp = x * cos(pomega) + y * sin(pomega),
               yp = -x * sin(pomega) + y * cos(pomega),

               // Rotate by the inclination angles.
               xsx = xp * sin(ix);

        pos[3 * i] = xp * cos(ix);
        pos[3 * i + 1] = yp * cos(iy) - xsx * sin(iy);
        pos[3 * i + 2] = yp * sin(iy) + xsx * cos(iy);
    }

    return info;
}

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "bart.h"
#include "kepler.h"

double occ_area (double r0, double p, double b)
{
    double r2, p2, b2, k1, k2, k3;

    if (b >= r0 + p) return 0.0;
    else if (b <= r0 - p) return M_PI * p * p;
    else if (b <= p - r0) return M_PI * r0 * r0;

    r2 = r0 * r0;
    p2 = p * p;
    b2 = b * b;

    k1 = acos(0.5 * (b2 + p2 - r2) / b / p);
    k2 = acos(0.5 * (b2 + r2 - p2) / b / r0);
    k3 = sqrt((p+r0-b) * (b+p-r0) * (b-p+r0) * (b+r0+p));

    return p2 * k1 + r2 * k2 - 0.5 * k3;
}

void ldlc (double p, int nbins, double *r, double *ir, int n, double *b,
           double *lam)
{
    int i, j;
    double *areas = malloc(nbins * sizeof(double));

    // First, compute the normalization constant by integrating over the face
    // of the star.
    double norm = ir[0] * r[0] * r[0];
    for (i = 1; i < nbins; ++i)
        norm += ir[i] * (r[i] * r[i] - r[i - 1] * r[i - 1]);
    norm *= M_PI;

    // Compute the fraction of un-occulted flux for each time sample.
    for (i = 0; i < n; ++i) {
        // Compute the array of occulted areas.
        for (j = 0; j < nbins; ++j) areas[j] = occ_area(r[j], p, b[i]);

        // Do the first order numerical integral over radial bins.
        lam[i] = areas[0] * ir[0];
        for (j = 1; j < nbins; ++j)
            lam[i] += ir[j] * (areas[j] - areas[j - 1]);
        lam[i] = 1.0 - lam[i] / norm;
    }

    // Clean up.
    free(areas);
}

int lightcurve (int n, double *t, double *flux,
                int nbin, double texp,
                double fstar, double mstar, double rstar, double iobs,
                int np, double *mass, double *r, double *a, double *t0,
                double *e, double *pomega, double *ix, double *iy,
                int nld, double *rld, double *ild)
{
    int i, j, info = 0;

    // Add extra time slices for integration over exposure time.
    double *ttmp = malloc(n * nbin * sizeof(double)),
           *ftmp = malloc(n * nbin * sizeof(double)),
           *tmp = malloc(n * nbin * sizeof(double)),
           *pos = malloc(3 * n * nbin * sizeof(double)),
           *b = malloc(n * nbin * sizeof(double));

    for (i = 0; i < n; ++i)
        for (j = 0; j < nbin; ++j)
            ttmp[i * nbin + j] = t[i] + texp * ((j + 0.5) / nbin - 0.5);

    // Initialize the light curve at the continuum.
    for (i = 0; i < n * nbin; ++i) ftmp[i] = fstar;

    // Loop over planets.
    for (i = 0; i < np; ++i) {
        // Compute the orbit.
        info = solve_orbit(n * nbin, ttmp, pos, mstar, mass[i], e[i], a[i],
                           t0[i], pomega[i],
                           (90 - iobs + ix[i]) / 180 * M_PI,
                           iy[i] / 180 * M_PI);

        // Did the solve fail?
        if (info != 0) goto cleanup;

        // Compute the impact parameter vector.
        for (j = 0; j < nbin * n; ++j) {
            b[j] = sqrt(pos[3 * j + 1] * pos[3 * j + 1]
                        + pos[3 * j + 2] * pos[3 * j + 2]) / rstar;
        }

        // HACK: deal with positions behind star.
        for (j = 0; j < nbin * n; ++j)
            if (pos[3 * j] <= 0) b[j] = 1.1 + r[i] / rstar;

        // Compute the fraction of occulted flux.
        ldlc(r[i] / rstar, nld, rld, ild, n * nbin, b, tmp);

        // Update the light curve.
        for (j = 0; j < nbin * n; ++j) ftmp[j] = ftmp[j] * tmp[j];
    }

    // "Integrate" over exposure time.
    for (i = 0; i < n; ++i) {
        flux[i] = 0.0;
        for (j = 0; j < nbin; ++j) flux[i] += ftmp[i * nbin + j] / nbin;
    }

cleanup:

    free(ttmp);
    free(ftmp);
    free(tmp);
    free(pos);
    free(b);

    return info;
}

void fast_lightcurve (int n, double *time, double *flux,
                      double rp, double period, double t0p,
                      int nld, double *rld, double *ild)
{
    int i;
    double a = pow(G_GRAV * period * period / 4 / M_PI / M_PI, 1. / 3.),
           phase, factor = 2 * M_PI / period,
           *b = malloc(n * sizeof(double));
    for (i = 0; i < n; ++i) {
        phase =  (time[i] - t0p) * factor;
        if (cos(phase) > 0) b[i] = a * sin(phase);
        else b[i] = 1.1 + rp;
    }
    ldlc(rp, nld, rld, ild, n, b, flux);
    free(b);
}

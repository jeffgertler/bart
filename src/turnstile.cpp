#include "george.h"
#include <stdlib.h>

typedef struct {

    int length;
    double *time, *flux, *ferr;

} Dataset;

Dataset *dataset_malloc (int n)
{
    Dataset *d = (Dataset*)malloc(sizeof(Dataset));
    d->length = n;
    d->time = (double*)malloc(n * sizeof(double));
    d->flux = (double*)malloc(n * sizeof(double));
    d->ferr = (double*)malloc(n * sizeof(double));
    return d;
}

void dataset_free (Dataset *d)
{
    free(d->time);
    free(d->flux);
    free(d->ferr);
    free(d);
}

double turnstile_marginalized_depth (int nsets, Dataset **datasets,
                                     double period, double epoch,
                                     double duration,
                                     double rmin, double rmax, double dr,
                                     double amp, double var)
{
    int info;
    double r, pars[2] = {amp, var}, *ftmp = malloc;
    George *gps = malloc();
    (2, pars, &gp_isotropic_kernel);

    // Compute the initial GP stuff.
    info = gp.compute(n, t, ferr);
    if (info != 0)
        return -INFINITY;

    // Loop over depth.
    for (r = rmin; r <= rmax; r += dr) {

    }
}

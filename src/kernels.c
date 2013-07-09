#include "kernels.h"
#include <math.h>

double gp_isotropic_kernel (double x1, double x2, int npars, double *pars)
{
    int i;
    double d = x1 - x2, chi2 = -0.5 * d * d, value = 0.0;
    for (i = 0; i + 1 < npars; i += 2)
        value += pars[i] * exp(chi2 / pars[i + 1]);
    return value;
}

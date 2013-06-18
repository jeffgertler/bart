#ifndef _KEPLER_H_
#define _KEPLER_H_

// Newton's constant in R_sun^3 M_sun^-1 days^-2.
#define G_GRAV 2945.4625385377644

int solve_orbit(int n, double *t, double *pos,
                double mstar, double mplanet, double e, double a, double t0,
                double pomega, double ix, double iy);

#endif
/* _KEPLER_H_ */

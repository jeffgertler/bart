#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as pl

import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(dirname)))
import bart


# Initialize the planetary system.
nbins, gamma1, gamma2 = 100, 0.39, 0.1
ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)
fstar = 1.0
mstar = 1.0
rstar = 1.0
iobs = 0.0
system = bart.BART(fstar, mstar, rstar, iobs, ldp)

# The parameters of the planets:
r = 0.0247
a = 6.47
e = 0.0
t0 = 2.38
pomega = 0.0
i = 89.76
system.add_planet(r, a, e, t0, pomega, i)

r = 0.02
a = 14.0
e = 0.0
t0 = 4.5
pomega = 0.0
i = 90.1
system.add_planet(r, a, e, t0, pomega, i)

N = 5000
t = 1.5 * 365.0 * np.sort(np.random.rand(N))
ferr = 8e-5 * np.random.rand(N)
flux = system.lightcurve(t) + ferr * np.random.randn(N)
pl.errorbar(t, flux, yerr=ferr, fmt=".k")
pl.savefig("raw.png")

system.fit(t, flux, ferr, pars=[u"fstar", u"t0", u"a", u"r"],
                        niter=100, thin=10, nburn=100, ntrim=1,
                        nwalkers=64)
system.plot_fit()
system.plot_triangle()

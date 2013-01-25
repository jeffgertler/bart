#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import pyfits
import numpy as np

import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(dirname)))
import bart


def load_data(fn="data.fits"):
    # Load the data.
    f = pyfits.open(os.path.join(dirname, "data.fits"))
    lc = np.array(f[1].data)
    f.close()

    time = lc["TIME"]
    flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

    t0 = int(np.median(time[~np.isnan(time)]))
    time = time - t0

    mu = np.median(flux)
    flux /= mu
    ferr /= mu * mu

    return (time, flux, ferr)


def default_ldp():
    # The limb-darkening parameters.
    nbins, gamma1, gamma2 = 100, 0.39, 0.1
    ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)

    if True:
        # Fit for the LDP with better spacing.
        ldp.bins = np.sqrt(np.linspace(0.0, 1.0, 15)[1:])
        rbins, ir = ldp.bins, ldp.intensity
        ir *= 1.0 / ir[0]
        ldp = bart.LimbDarkening(rbins, ir)

    return ldp


def build_model():
    i = 89.76
    T = 3.21346
    planet = bart.Planet(r=0.0247, a=6.47, t0=2.38)

    planet.parameters.append(bart.LogParameter("$r$", "r"))
    planet.parameters.append(bart.LogParameter("$a$", "a"))
    planet.parameters.append(bart.LogParameter("$t0$", "t0"))

    star = bart.Star(mass=planet.get_mstar(T))
    system = bart.PlanetarySystem(star, iobs=i)
    system.add_planet(planet)

    # Fit it.
    t, f, ferr = load_data()
    system.fit(t, f, ferr, niter=5000, thin=500, nburn=1000, ntrim=1,
               nwalkers=64)

    assert 0
    system.plot_fit()
    system.plot_triangle()


if __name__ == "__main__":
    build_model()

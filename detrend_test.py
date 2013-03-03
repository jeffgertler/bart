#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
import matplotlib.pyplot as pl

import george

import bart
from bart.dataset import KeplerDataset
from bart import kepler


def robust_polyfit(x, y, yerr=None, order=3, Q=1.5):
    if yerr is None:
        yerr = np.ones_like(y)
    A = np.vstack([x ** i for i in range(order + 1)[::-1]]).T
    inds = np.ones_like(x, dtype=bool)
    for i in range(10):
        a = np.linalg.lstsq(A[inds, :] / yerr[inds][:, None],
                            y[inds] / yerr[inds])[0]
        p = np.poly1d(a)
        delta = (y - p(x)) ** 2 / yerr ** 2
        sigma = np.median(delta[inds])
        inds = delta < Q * sigma
    return p, inds


def fit_gp(x, y, yerr=None):
    p = george.GaussianProcess([0.5, 1.])
    p.fit(x, y, yerr=yerr)
    return p


if __name__ == "__main__":
    api = bart.kepler.API()
    fns = api.data("10874614").fetch_all("kepler6data")

    rstar = 1.391  # +0.017 -0.034 R_sun
    Teff = 5647.
    logg = 4.24
    feh = 0.34

    P = 3.234723  # ± 0.000017 days
    a = 7.05  # +0.11 -0.06 R_*
    r = 0.09829  # +0.00014 -0.00050 R_*
    i = 86.8  # ± 0.3 degrees
    mass = 0.669 * 9.5492e-4  # Solar masses.

    # The reference "transit" time.
    t0 = 1.795  # 0.28  # Found by eye.

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0, mass=mass)

    # Set up the star.
    ldp = kepler.fiducial_ldp(Teff, logg, feh, bins=15, alpha=0.5)
    star = bart.Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)

    # Set up the system.
    system = bart.PlanetarySystem(star, iobs=i, rv0=-15.0,
                                  basepath="kepler6data")
    system.add_planet(planet)

    for i, fn in enumerate(fns):
        pl.clf()
        ds = KeplerDataset(fn)

        x, y, yerr = ds.time, ds.flux, ds.ferr
        # y = kepler.spline_detrend(x, y, yerr=yerr)
        # y = spline_detrend(x, y, yerr=yerr)

        y -= system.lightcurve(x, texp=ds.texp)
        print(len(y))
        N = 500
        gp = fit_gp(x[:N], y[:N], yerr=yerr[:N])
        print("fitted")
        print(gp.evaluate())
        m, v = gp.predict(x, full_cov=False)
        print(m.shape, v.shape)

        pl.plot(x, y, "+k")
        pl.plot(x, m, "--r")
        pl.plot(x, m + np.sqrt(v), ":r")
        pl.plot(x, m - np.sqrt(v), ":r")
        pl.title(fn.replace("_", r"\_"))
        pl.savefig("detrend_test/{0:03d}.png".format(i))

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
import emcee
import numpy as np
import matplotlib.pyplot as pl
import kplr
from kplr.ld import get_quad_coeffs

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

import bart
from bart import _george

client = kplr.API()

# Query the KIC and get some parameters.
kic = client.star(2301306)
teff, logg, feh = kic.kic_teff, kic.kic_logg, kic.kic_feh
assert teff is not None

# Get the limb darkening law.
mu1, mu2 = get_quad_coeffs(teff, logg=logg, feh=feh)
bins = np.linspace(0, 1, 50)[1:] ** 0.5
ldp = bart.ld.QuadraticLimbDarkening(mu1, mu2).histogram(bins)

# Build the star object.
star = bart.Star(ldp=ldp)

# Set up the planet.
period = 50.0
size = 0.05
epoch = 10.0
a = star.get_semimajor(period)
b = 0.5
incl = np.degrees(np.arctan2(a, b))
planet = bart.Planet(size, a, t0=epoch)

# Set up the system.
ps = bart.PlanetarySystem(star, iobs=incl)
ps.add_planet(planet)

# Load the data and inject into each transit.
lcs = kic.get_light_curves(short_cadence=False, fetch=False)
lc = lcs[6]

with lc.open() as f:
    # The light curve data are in the first FITS HDU.
    hdu_data = f[1].data

    # Load the data columns.
    time = hdu_data["time"]
    time -= np.min(time[np.isfinite(time)])
    sap_flux = hdu_data["sap_flux"]
    sap_ferr = hdu_data["sap_flux_err"]
    quality = hdu_data["sap_quality"]

    # Cut out a small chunk of data.
    inds = (time < 30.0) * np.isfinite(time) * np.isfinite(sap_flux)
    time = time[inds]
    sap_flux = sap_flux[inds]
    sap_ferr = sap_ferr[inds]

# Inject a transit.
sap_flux *= ps.lightcurve(time)

# Normalize the data.
mu = np.median(sap_flux)
sap_flux /= mu
sap_ferr /= mu


def lnlike(p):
    planet.t0, planet.size = p[0], np.exp(p[1])
    model = sap_flux / ps.lightcurve(time)
    return _george.lnlikelihood(time, model, sap_ferr, 1.0, 3.0)


if __name__ == "__main__":
    nwalkers = 10
    p0 = zip(epoch + 0.01 * np.random.randn(nwalkers),
             np.log(np.abs(size * (1 + 0.01 * np.random.randn(nwalkers)))))

    sampler = emcee.EnsembleSampler(nwalkers, len(p0[0]), lnlike,
                                    threads=nwalkers)
    fn = "samples.txt"
    with open(fn, "w") as f:
        f.write("# t0 ln(size) ln(prob)\n")
    for pos, lnprob, state in sampler.sample(p0, iterations=10,
                                             storechain=False):
        with open(fn, "a") as f:
            for p, lp in zip(pos, lnprob):
                f.write("{0} {1} {2}\n".format(p[0], p[1], lp))

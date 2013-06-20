#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
import emcee
import triangle
import matplotlib.pyplot as pl
import numpy as np
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
period = 300.0
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

# Loop over the datasets and read in the data.
time, flux, ferr, quality = [np.array([]) for i in range(4)]
for lc in lcs:
    with lc.open() as f:
        # The light curve data are in the first FITS HDU.
        hdu_data = f[1].data
        time_, flux_, ferr_, quality = [hdu_data[k] for k in ["time",
                                                              "sap_flux",
                                                              "sap_flux_err",
                                                              "sap_quality"]]

    # Mask the missing data.
    mask = (np.isfinite(time_) * np.isfinite(flux_) * np.isfinite(ferr_)
            * (quality == 0))
    time_, flux_, ferr_ = [v[mask] for v in [time_, flux_, ferr_]]

    # Inject the transit.
    flux_ *= ps.lightcurve(time_)

    # Normalize the data.
    mu = np.median(flux_)

    time = np.append(time, time_)
    flux = np.append(flux, flux_ / mu)
    ferr = np.append(ferr, ferr_ / mu)

# Cut out only data around transits.
inds = np.abs((time - epoch + 0.5 * period) % period - 0.5 * period) < 5
time, flux, ferr = time[inds], flux[inds], ferr[inds]

pl.plot(time % period, flux, ".k")
pl.savefig("data.png")


def lnlike(p):
    if not 0 < p[1] < 1 or p[2] <= 1 or not 0 <= p[3] < 1:
        return -np.inf
    planet.t0, planet.r, planet.a, b = p
    ps.iobs = np.degrees(np.arctan2(planet.a, b))
    model = flux / ps.lightcurve(time) - 1
    return _george.lnlikelihood(time, model, ferr, 1.0, 3.0)


if __name__ == "__main__":
    nwalkers = 10
    p0 = zip(epoch + 0.001 * np.random.randn(nwalkers),
             np.abs(size * (1 + 0.01 * np.random.randn(nwalkers))),
             np.abs(a * (1 + 1e-4 * np.random.randn(nwalkers))),
             np.abs(b * (1 + 0.01 * np.random.randn(nwalkers))))

    sampler = emcee.EnsembleSampler(nwalkers, len(p0[0]), lnlike,
                                    threads=nwalkers)
    fn = "samples.txt"
    with open(fn, "w") as f:
        f.write("# t0 r/R a/R b/R ln(prob)\n")
    for pos, lnprob, state in sampler.sample(p0, iterations=500,
                                             storechain=False):
        with open(fn, "a") as f:
            for p, lp in zip(pos, lnprob):
                f.write("{0} {1}\n".format(
                    " ".join(map("{0}".format, p)), lp))

    print("Acceptance fraction: {0}"
          .format(np.mean(sampler.acceptance_fraction)))

    samples = np.loadtxt(fn)
    figure = triangle.corner(samples)
    figure.savefig("triangle.png")

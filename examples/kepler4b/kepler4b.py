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
from bart.results import ResultsProcess


def load_data(fn="data.fits"):
    # Load the data.
    f = pyfits.open(os.path.join(dirname, "data.fits"))
    lc = np.array(f[1].data)
    f.close()

    time = lc["TIME"]
    flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]

    t0 = int(np.median(time[~np.isnan(time)]))
    time = time - t0

    mu = np.median(flux[~np.isnan(flux)])
    flux /= mu
    ferr /= mu

    return (time, flux, ferr)


def default_ldp():
    # The limb-darkening parameters.
    nbins, gamma1, gamma2 = 100, 0.39, 0.1
    ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)
    return ldp


def build_model():
    # Some basic known parameters about the system.
    i = 89.76
    T = 3.21346

    # Set up the planet based on the Kepler team results for this object.
    planet = bart.Planet(r=0.0247, a=6.47, t0=2.38, e=0.01, pomega=0.001)

    # Add some fit parameters to the planet.
    planet.parameters.append(bart.LogParameter("$r$", "r"))
    planet.parameters.append(bart.LogParameter("$a$", "a"))
    planet.parameters.append(bart.LogParameter("$t0$", "t0"))
    planet.parameters.append(bart.EccentricityParameters())

    # A star needs to have a mass and a limb-darkening profile.
    star = bart.Star(mass=planet.get_mstar(T), ldp=default_ldp())

    # Set up the planetary system.
    system = bart.PlanetarySystem(star, iobs=i)

    # Add the planet to the system.
    system.add_planet(planet)

    # Read in the Kepler light curve.
    t, f, ferr = load_data()

    # Plot initial guess.
    # pl.plot(t % T, f, ".k", alpha=0.3)
    # ts = np.linspace(0, T, 1000)
    # pl.plot(ts, system.lightcurve(ts))
    # pl.savefig("initial.png")

    # Do the fit.
    system.fit((t, f, ferr), 1000, thin=10, burnin=[], nwalkers=16)

    # Plot the results.
    results = ResultsProcess("./mcmc.h5")
    results.corner_plot()
    results._lc_plot(["lightcurves", 0])


if __name__ == "__main__":
    build_model()

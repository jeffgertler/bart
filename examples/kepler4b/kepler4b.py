#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import pyfits
import numpy as np

import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(dirname)))
import bart

from bart.parameters.base import LogParameter, Parameter
from bart.parameters.star import LimbDarkeningParameters
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
    nbins, gamma1, gamma2 = 10, 0.39, 0.1
    ldp = bart.QuadraticLimbDarkening(nbins, gamma1, gamma2)
    return ldp


def build_model():
    # Some basic known parameters about the system.
    i = 89.76
    T = 3.21346

    # Set up the planet based on the Kepler team results for this object.
    planet = bart.Planet(r=0.0247, a=6.471, t0=2.38)

    # Add some fit parameters to the planet.
    # planet.parameters.append(LogParameter("$r$", "r"))
    # planet.parameters.append(LogParameter("$a$", "a"))
    # planet.parameters.append(LogParameter("$t_0$", "t0"))

    # A star needs to have a mass and a limb-darkening profile.
    star = bart.Star(mass=planet.get_mstar(T), ldp=default_ldp())
    # star.parameters.append(Parameter("$f_\star$", "flux"))
    star.parameters.append(LimbDarkeningParameters(star.ldp.bins))

    # Set up the planetary system.
    system = bart.PlanetarySystem(star, iobs=i)

    # Add the planet to the system.
    system.add_planet(planet)

    # Read in the Kepler light curve.
    t, f, ferr = load_data()

    # Plot initial guess.
    import matplotlib.pyplot as pl
    pl.plot(t % T, f, ".k", alpha=0.3)
    ts = np.linspace(0, T, 1000)
    pl.plot(ts, system.lightcurve(ts))
    pl.savefig("initial.png")
    # assert 0

    # Do the fit.
    bp = "."
    system.fit((t, f, ferr), 100, thin=10, burnin=[200], nwalkers=32,
               basepath=bp)

    # Plot the results.
    results = ResultsProcess(basepath=bp)
    results.corner_plot()
    # results.lc_plot()
    # results.time_plot()


if __name__ == "__main__":
    build_model()

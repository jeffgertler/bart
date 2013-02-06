#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))))
import bart
from bart import kepler
from bart.results import Column
from bart.parameters.base import Parameter, LogParameter
from bart.parameters.star import RelativeLimbDarkeningParameters

import numpy as np
import matplotlib.pyplot as pl


# Reproducible Science.™
np.random.seed(100)


class CosParameter(Parameter):

    def getter(self, obj):
        return np.cos(np.radians(obj.iobs))

    def setter(self, obj, val):
        obj.iobs = np.degrees(np.arccos(val))

    def sample(self, obj, std=1e-5, size=1):
        return np.cos(np.radians(obj.iobs * (1 + std * np.random.randn(size))))


def main():
    # Initial physical parameters from:
    #  http://kepler.nasa.gov/Mission/discoveries/kepler6b/
    #  http://arxiv.org/abs/1001.0333

    # mstar = 1.209  # +0.044 -0.038 M_sun
    rstar = 1.391  # +0.017 -0.034 R_sun
    Teff = 5647.
    logg = 4.24
    feh = 0.34

    P = 3.234723  # ± 0.000017 days
    # E = 2454954.48636  # ± 0.00014 HJD

    # a = 9.80650134 / rstar  # Mine
    a = 7.05  # +0.11 -0.06 R_*

    # r = 0.12342658 / rstar  # Mine.
    r = 0.09829  # +0.00014 -0.00050 R_*

    # i = 90.0  # Mine.
    i = 86.8  # ± 0.3 degrees

    # Compute the reference transit time.
    t0 = 1.795  # Found by eye.

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0)

    # Set up the star.
    ldp = kepler.fiducial_ldp(Teff, logg, feh, bins=15, alpha=0.5)
    star = bart.Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)

    # Set up the system.
    system = bart.PlanetarySystem(star, iobs=i, basepath="kepler6")
    system.add_planet(planet)

    # Fit parameters.
    planet.parameters.append(Parameter(r"$r$", "r"))
    planet.parameters.append(LogParameter(r"$a$", "a"))
    planet.parameters.append(Parameter(r"$t_0$", "t0"))

    system.parameters.append(CosParameter(r"$i$", "iobs"))

    star.parameters.append(RelativeLimbDarkeningParameters(star.ldp.bins,
                                                   star.ldp.intensity,
                                                   eta=0.01))

    # Get the data.
    api = kepler.API()
    print("Downloading the data files.")
    data_files = api.data("10874614").fetch_all(basepath="kepler6/data")

    # Read in the data.
    time, flux, ferr = np.array([]), np.array([]), np.array([])
    for i, fn in enumerate(data_files):
        if "slc" in fn:
            t, f, fe = kepler.load(fn)
            time = np.append(time, t)
            flux = np.append(flux, f)
            ferr = np.append(ferr, fe)
            break

            # Do the fit.
            # system.vector = vector0

    # system.fit((time, flux, ferr), 1, thin=1, burnin=[], nwalkers=64)
    # system.fit((time, flux, ferr), 2000, thin=10, burnin=[], nwalkers=64)

    # Plot the results.
    print("Plotting results")
    results = system.results(thin=1, burnin=50)
    results.latex([
            Column(r"$P\,[\mathrm{days}]$",
                   lambda s: s.planets[0].get_period(s.star.mass)),
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0\,[\mathmr{days}]$", lambda s: s.planets[0].t0),
            Column(r"$i\,[\mathrm{deg}]$", lambda s: s.iobs),
        ])

    results.lc_plot()
    results.ldp_plot(fiducial=kepler.fiducial_ldp(Teff, logg, feh))
    results.time_plot()
    results.corner_plot([
            Column(r"$a$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
            Column(r"$i$", lambda s: s.iobs),
        ])


if __name__ == "__main__":
    main()

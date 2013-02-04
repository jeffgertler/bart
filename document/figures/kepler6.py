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
from bart.parameters.star import LimbDarkeningParameters

import numpy as np
# import matplotlib.pyplot as pl


# Reproducible Science.™
np.random.seed(123)


def main():
    # Initial physical parameters from:
    #  http://kepler.nasa.gov/Mission/discoveries/kepler6b/
    #  http://arxiv.org/abs/1001.0333
    P = 3.234723  # ± 0.000017 days
    # E = 2454954.48636  # ± 0.00014 HJD
    a = 7.05  # +0.11 -0.06 R_*
    r = 0.09829  # +0.00014 -0.00050 R_*
    i = 86.8  # ± 0.3 degrees

    # mstar = 1.209  # +0.044 -0.038 M_sun
    rstar = 1.391  # +0.017 -0.034 R_sun

    # Compute the reference transit time.
    t0 = 1.8

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0)
    planet.parameters.append(LogParameter(r"$r$", "r"))
    planet.parameters.append(LogParameter(r"$a$", "a"))
    planet.parameters.append(LogParameter(r"$t_0$", "t0"))

    # Set up the star.
    rs = np.linspace(0, 1, 15) ** 0.5
    ldp = kepler.fiducial_ldp(rs[1:])
    star = bart.Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)
    star.parameters.append(LogParameter(r"$f_\star$", "flux"))
    star.parameters.append(LimbDarkeningParameters(star.ldp.bins))

    # Set up the system.
    system = bart.PlanetarySystem(star, iobs=i, basepath="kepler6")
    system.parameters.append(Parameter(r"$i$", "iobs"))
    system.add_planet(planet)

    # Get the data.
    api = kepler.API()
    data_files = api.data("10874614").fetch_all(basepath="kepler6/data")

    # Read in the data.
    time, flux, ferr = np.array([]), np.array([]), np.array([])
    for fn in data_files:
        if "llc" in fn:
            t, f, fe = kepler.load(fn)
            time = np.append(time, t)
            flux = np.append(flux, f)
            ferr = np.append(ferr, fe)

    # Do the fit.
    system.fit((time, flux, ferr), 10000, thin=200, burnin=[500], nwalkers=64)

    # Plot the results.
    results = system.results
    results.lc_plot()
    results.ldp_plot()
    results.time_plot()
    results.corner_plot([
            Column(r"$a$", lambda s: s.planets[0].a),
            Column(r"$r$", lambda s: s.planets[0].r),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
            Column(r"$i$", lambda s: s.iobs),
        ])


if __name__ == "__main__":
    main()

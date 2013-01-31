#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np

import os
import sys
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(dirname)))
import bart

from bart import kepler

from bart.parameters.base import LogParameter, Parameter
from bart.parameters.star import LimbDarkeningParameters
from bart.results import Column


def build_model():
    # Some basic known parameters about the system.
    bp = "."
    i = 89.76
    T = 3.21346

    # Set up the planet based on the Kepler team results for this object.
    planet = bart.Planet(r=0.0247, a=6.471, t0=2.38)

    # Add some fit parameters to the planet.
    planet.parameters.append(LogParameter("$r$", "r"))
    planet.parameters.append(LogParameter("$a$", "a"))
    planet.parameters.append(Parameter("$i$", "ix"))
    planet.parameters.append(LogParameter("$t_0$", "t0"))

    # A star needs to have a mass and a limb-darkening profile.
    star = bart.Star(mass=planet.get_mstar(T),
                     ldp=kepler.fiducial_ldp(np.linspace(0, 1, 15) ** 0.25))
    star.parameters.append(LimbDarkeningParameters(star.ldp.bins))

    # Set up the planetary system.
    system = bart.PlanetarySystem(star, iobs=i, basepath=bp)

    # Add the planet to the system.
    system.add_planet(planet)

    # Read in the Kepler light curve.
    t, f, ferr = kepler.load("data.fits")

    # Plot initial guess.
    import matplotlib.pyplot as pl
    pl.plot(t % T, f, ".k", alpha=0.3)
    ts = np.linspace(0, T, 1000)
    pl.plot(ts, system.lightcurve(ts))
    pl.savefig("initial.png")
    # assert 0

    # Do the fit.
    system.fit((t, f, ferr), 5000, thin=500, burnin=[200], nwalkers=50)

    # Plot the results.
    results = system.results

    # Make a "corner" plot of the parameters.
    results.corner_plot([
            Column(r"$a$", lambda s: s.planets[0].a),
            Column(r"$r$", lambda s: s.planets[0].r),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
        ])

    # Plot the time series of the parameters.
    results.time_plot()

    # Plot samples of the limb darkening profile.
    results.ldp_plot()

    # Plot samples of the light curve.
    results.lc_plot()


if __name__ == "__main__":
    build_model()

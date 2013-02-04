#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A BART example that can fit the light curve for any Kepler confirmed planets.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))))
import bart
from bart import kepler
from bart.results import Column
from bart.parameters.base import LogParameter
from bart.parameters.star import LimbDarkeningParameters

import numpy as np
import matplotlib.pyplot as pl


def main(name="KEPLER-4 b"):
    # Resolve the planet name.
    api = kepler.API()
    planets = api.planets(kepler_name=name)
    if planets is None:
        print("No planet with the name '{0}'. Try: 'KEPLER-4 b'.".format(name))
        sys.exit(1)

    # Find all of the planets in the same system.
    kepid = planets[0]["Kepler ID"]
    planets = api.kois(kepid=kepid)
    if planets is None:
        print("Something went wrong.")
        sys.exit(2)

    # Fetch the data.
    bp = "data"
    data_files = api.data(kepid).fetch_all(basepath=bp)

    # Open all the files and combine them.
    # NOTE: the median flux is divided out of the light curves on a
    # file-by-file basis. This seems to be reasonable for now.
    time, flux, ferr = np.array([]), np.array([]), np.array([])
    for fn in data_files:
        if "llc" in fn:
            t, f, fe = kepler.load(fn)
            time = np.append(time, t)
            flux = np.append(flux, f)
            ferr = np.append(ferr, fe)

    # Set up the system.
    rs = np.linspace(0, 1, 15) ** 0.5
    ldp = kepler.fiducial_ldp(rs[1:])
    star = bart.Star(radius=2.0,  # float(planets[0]["Stellar Radius"]),
                     ldp=ldp)
    star.parameters.append(LimbDarkeningParameters(star.ldp.bins))

    system = bart.PlanetarySystem(star, iobs=90.0)

    for p in planets:
        # Earth radii to Solar radii and AU to Solar radii from ``astropy``.
        r = 2 * 0.0247  # 0.009170471080131358 * float(p["Planet Radius"])  # / star.radius
        a = 2 * 6.471   # 215.09151684811675 * float(p["Semi-major Axis"])  # / star.radius
        print(r, a)
        planet = bart.Planet(r=r, a=a, t0=0.0)

        # Fit parameters.
        planet.parameters.append(LogParameter("$r$", "r"))
        planet.parameters.append(LogParameter("$a$", "a"))
        # planet.parameters.append(Parameter("$i$", "ix"))
        planet.parameters.append(LogParameter("$t_0$", "t0"))

        # Add the planet.
        system.add_planet(planet)

    # Find the stellar mass from the periods.
    star.mass = np.mean([system.planets[i].get_mstar(float(p["Period"]))
                         for i, p in enumerate(planets)])

    # Find the epoch value for each planet.
    for i, p in enumerate(system.planets):
        period = p.get_period(star.mass)
        min_chi2 = None
        for t0 in np.linspace(1.35, 1.5, 10):
            p.t0 = t0
            chi2 = np.median(((system.lightcurve(time) - flux) / ferr) ** 2)
            if min_chi2 is None or chi2 < min_chi2[0]:
                min_chi2 = (chi2, t0)
            print(t0, chi2)

        print(min_chi2)
        p.t0 = 1.49  # min_chi2[1]

        t = np.linspace(0, period, 5000)
        pl.plot(time % period, flux, ".k", alpha=0.1)
        pl.plot(t, system.lightcurve(t))
        pl.savefig("blah.{0}.png".format(i))

    assert 0

    system.planets[0].t0 = 1.49

    system.fit((time, flux, ferr), 200, thin=20, burnin=[], nwalkers=64)
    results = system.results
    results.lc_plot()
    results.ldp_plot()
    results.time_plot()
    results.corner_plot([
            Column(r"$a$", lambda s: s.planets[0].a),
            Column(r"$r$", lambda s: s.planets[0].r),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
        ])


if __name__ == "__main__":
    main()

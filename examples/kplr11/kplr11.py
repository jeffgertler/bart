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
from bart.parameters.base import LogParameter


def main():
    t, f, ferr = kepler.load("data.fits")

    star = bart.Star(mass=0.95, radius=1.1,
                     ldp=kepler.fiducial_ldp(np.linspace(0, 1, 15) ** 0.25))

    k11b = bart.Planet(t0=4.55, r=0.01638, a=star.get_semimajor(10.30375))
    k11c = bart.Planet(t0=7.9, r=0.02615, a=star.get_semimajor(13.02502))
    k11d = bart.Planet(t0=12.3, r=0.02861, a=star.get_semimajor(22.68719))
    k11e = bart.Planet(t0=26.2, r=0.03791, a=star.get_semimajor(31.99590))
    # k11f = bart.Planet(t0=12.3, r=0.02171, a=star.get_semimajor(46.68876))
    # k11g = bart.Planet(t0=12.3, r=0.03087, a=star.get_semimajor(22.68719))

    # The system.
    system = bart.PlanetarySystem(star, iobs=89.0)
    system.add_planet(k11b)
    system.add_planet(k11c)
    system.add_planet(k11d)
    system.add_planet(k11e)

    # import matplotlib.pyplot as pl
    # P = 46.68876
    # pl.plot(t % P, f, ".k", alpha=0.1)
    # ts = np.linspace(0, P, 1000)
    # pl.plot(ts, system.lightcurve(ts), "r")
    # pl.savefig("initial.png")
    # assert 0

    for planet in system.planets:
        planet.parameters.append(LogParameter("$r$", "r"))
        planet.parameters.append(LogParameter("$a$", "a"))
        planet.parameters.append(LogParameter("$t_0$", "t0"))

    system.fit((t, f, ferr), 200, thin=10, burnin=[100], nwalkers=64)

    # Plot the results.
    results = system.results

    # Make a "corner" plot of the parameters.
    # results.corner_plot([
    #         Column(r"$a$", lambda s: s.planets[0].a),
    #         Column(r"$r$", lambda s: s.planets[0].r),
    #         Column(r"$t_0$", lambda s: s.planets[0].t0),
    #     ])

    results.time_plot()
    results.lc_plot()


if __name__ == "__main__":
    main()

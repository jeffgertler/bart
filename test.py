#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from bart import Planet, Star, PlanetarySystem
from bart import kepler
import numpy as np
import matplotlib.pyplot as pl


def main():
    rstar = 1.391  # +0.017 -0.034 R_sun
    Teff = 5647.
    logg = 4.24
    feh = 0.34

    P = 3.234723  # ± 0.000017 days
    a = 7.05  # +0.11 -0.06 R_*
    r = 0.09829  # +0.00014 -0.00050 R_*
    i = 86.8  # ± 0.3 degrees
    t0 = 1.795

    # Set up the planet.
    planet = Planet(r=r * rstar, a=a * rstar, t0=t0)

    # Set up the star.
    ldp = kepler.fiducial_ldp(Teff, logg, feh, bins=15, alpha=0.5)
    star = Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)

    # Set up the system.
    system = PlanetarySystem(star, iobs=i, basepath="kepler6")
    system.add_planet(planet)

    # Plot a transit.
    t = np.linspace(0, P, 5000)
    pl.plot(t, system.lightcurve(t))
    pl.savefig("lctest.png")


if __name__ == "__main__":
    main()

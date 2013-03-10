#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


import numpy as np
import matplotlib.pyplot as pl

import bart
from bart.kepler import fiducial_ldp


# The Star.
ldp = fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50)
star = bart.Star(mass=1.209, radius=1.391, ldp=ldp)

# The Planet.
a = 7.05 * star.radius
Rp = 0.09829 * star.radius
planet = bart.Planet(a=a, r=Rp)

print(planet.get_period(star.mass))

# The system.
kepler6 = bart.PlanetarySystem(star, iobs=86.8)
kepler6.add_planet(planet)

# Plot the model light curve.
t = np.linspace(-0.2, 0.2, 5000)
pl.plot(t, kepler6.lightcurve(t), "k", lw=2)
pl.xlim(-0.2, 0.2)
pl.ylim(0.9875, 1.001)
pl.xlabel("Time Since Transit")
pl.ylabel("Relative Flux")
pl.savefig("model_building.png")

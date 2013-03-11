#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


import numpy as np
import matplotlib.pyplot as pl

import bart
from bart.kepler import fiducial_ldp

np.random.seed(123)


# The Star.
ldp = fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50)
star = bart.Star(mass=1.209, radius=1.391, ldp=ldp)

# The Planet.
a = 7.05 * star.radius
Rp = 0.09829 * star.radius
planet = bart.Planet(a=a, r=Rp)
period = planet.get_period(star.mass)

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

# Generate synthetic data.
# Long cadence.
time = np.arange(0, 90., 1766 / (60. * 60. * 24.))
folded_time = (time + 0.5 * period) % period - 0.5 * period
model_flux = kepler6.lightcurve(time, texp=1626.)

# Add noise.
ferr = 1.5e-3 * np.random.rand(len(model_flux))
flux = model_flux + ferr * np.random.randn(len(model_flux))

# Plot.
pl.clf()
pl.subplot(211)
pl.plot(folded_time, flux, ".k")
pl.xlim(-0.2, 0.2)
pl.ylim(0.981, 1.01)
pl.gca().set_xticklabels([])
pl.annotate("long cadence", xy=[0, 0], xycoords="axes fraction",
            xytext=[8, 8], textcoords="offset points")
pl.ylabel("Relative Flux")

# Short cadence.
time = np.arange(0, 60., 58.9 / (60. * 60. * 24.))
folded_time = (time + 0.5 * period) % period - 0.5 * period
model_flux = kepler6.lightcurve(time, texp=54.2)

# Add noise.
ferr = 3e-3 * np.random.rand(len(model_flux))
flux = model_flux + ferr * np.random.randn(len(model_flux))

# Plot.
pl.subplot(212)
pl.plot(folded_time, flux, ".k")
pl.xlim(-0.2, 0.2)
pl.ylim(0.981, 1.01)
pl.annotate("short cadence", xy=[0, 0], xycoords="axes fraction",
            xytext=[8, 8], textcoords="offset points")
pl.xlabel("Time Since Transit")
pl.ylabel("Relative Flux")
pl.savefig("model_building_data.png")

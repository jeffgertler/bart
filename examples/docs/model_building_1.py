#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)


import numpy as np
import matplotlib.pyplot as pl

import bart
from bart.dataset import Dataset, RVDataset
from bart.kepler import fiducial_ldp, TIME_ZERO

np.random.seed(123)


def generate_synthetic_data():
    # The Star.
    ldp = fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50)
    star = bart.Star(mass=1.209, radius=1.391, ldp=ldp)

    # The Planet.
    a = 7.05 * star.radius
    Rp = 0.09829 * star.radius
    planet = bart.Planet(a=a, r=Rp, mass=6.388e-4)

    # The system.
    kepler6 = bart.PlanetarySystem(star, iobs=86.8, rv0=-49.14)
    kepler6.add_planet(planet)

    # Long cadence.
    lc_time = np.arange(0, 90., 1766 / (60. * 60. * 24.))
    lc_flux = kepler6.lightcurve(lc_time, texp=1626.)
    lc_err = 1.5e-3 * np.random.rand(len(lc_flux))
    lc_flux = lc_flux + lc_err * np.random.randn(len(lc_flux))

    # Short cadence.
    sc_time = np.arange(0, 60., 58.9 / (60. * 60. * 24.))
    sc_flux = kepler6.lightcurve(sc_time, texp=54.2)
    sc_err = 3e-3 * np.random.rand(len(sc_flux))
    sc_flux = sc_flux + sc_err * np.random.randn(len(sc_flux))

    # Radial velocity.
    rv_time = 100 * np.sort(np.random.rand(15)) + TIME_ZERO
    rv = kepler6.radial_velocity(rv_time)
    rv_err = 5 + 2 * np.random.randn(len(rv_time))
    rv += rv_err * np.random.randn(len(rv_time))

    return (kepler6,
            Dataset(lc_time, lc_flux, lc_err, 1626.),
            Dataset(sc_time, sc_flux, sc_err, 54.2),
            RVDataset(rv_time, rv, rv_err))


if __name__ == "__main__":
    # Generate the synthetic data.
    kepler6, lc, sc, rv = generate_synthetic_data()
    period = kepler6.planets[0].get_period(kepler6.star.mass)

    # Make plots.
    # Model light curve.
    t = np.linspace(-0.2, 0.2, 5000)
    pl.plot(t, kepler6.lightcurve(t), "k", lw=2)
    pl.xlim(-0.2, 0.2)
    pl.ylim(0.9875, 1.001)
    pl.xlabel("Time Since Transit")
    pl.ylabel("Relative Flux")
    pl.savefig("model_building.png")

    # Synthetic dataset plot.
    # - Long cadence.
    pl.clf()
    pl.subplot(211)
    lc_folded = (lc.time + 0.5 * period) % period - 0.5 * period
    pl.plot(lc_folded, lc.flux, ".k")
    pl.xlim(-0.2, 0.2)
    pl.ylim(0.981, 1.01)
    pl.gca().set_xticklabels([])
    pl.annotate("long cadence", xy=[0, 0], xycoords="axes fraction",
                xytext=[8, 8], textcoords="offset points")
    pl.ylabel("Relative Flux")

    # - Short cadence.
    pl.subplot(212)
    sc_folded = (sc.time + 0.5 * period) % period - 0.5 * period
    pl.plot(sc_folded, sc.flux, ".k")
    pl.xlim(-0.2, 0.2)
    pl.ylim(0.981, 1.01)
    pl.annotate("short cadence", xy=[0, 0], xycoords="axes fraction",
                xytext=[8, 8], textcoords="offset points")
    pl.xlabel("Time Since Transit")
    pl.ylabel("Relative Flux")
    pl.savefig("model_building_data.png")

    # Plot the radial velocity.
    pl.clf()
    rv_folded = (rv.time + 0.5 * period) % period - 0.5 * period
    pl.errorbar(rv_folded, rv.rv, yerr=rv.rverr, fmt=".k")
    t = np.linspace(0.5 * period, 1.5 * period, 1000) + TIME_ZERO
    pl.plot(t - period - TIME_ZERO, kepler6.radial_velocity(t))
    pl.gca().axhline(kepler6.rv0, color="k", ls="dashed")
    pl.xlim(-0.5 * period, 0.5 * period)
    pl.savefig("model_building_rv_data.png")

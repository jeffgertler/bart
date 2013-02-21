#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import bart
from bart import kepler
from bart.dataset import KeplerDataset

import numpy as np
import matplotlib.pyplot as pl


def main():
    api = kepler.API()
    data = api.kois(kepoi="686.01")[0]
    files = api.data(data["Kepler ID"]).fetch_all("data")

    rstar = float(data["Stellar Radius"])
    rho = 1.02
    teff = float(data["Teff"])
    logg = float(data["log(g)"])
    feh = 0.0

    a = float(data["a/R"]) * rstar
    r = float(data["r/R"]) * rstar
    e = 0.62  # from Dawson & Johnson
    i = 90.0
    P = float(data["Period"])
    t0 = 0.0

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0, e=e)

    # Set up the star.
    ldp = kepler.fiducial_ldp(teff, logg, feh, bins=15, alpha=0.5)
    star = bart.Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)

    # Set up the system.
    system = bart.PlanetarySystem(star, iobs=i)
    system.add_planet(planet)

    # Add the datasets.
    for fn in files:
        system.add_dataset(KeplerDataset(fn))

    [pl.plot(d.time % P, d.flux, ".k", alpha=0.1) for d in system.datasets]
    ts = np.linspace(0, P, 5000)
    pl.plot(ts, system.lightcurve(ts))
    pl.savefig("initial_lc.png")


if __name__ == "__main__":
    main()

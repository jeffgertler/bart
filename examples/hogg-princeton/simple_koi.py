#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import bart
from bart import kepler
from bart.dataset import KeplerDataset
from bart.parameters.base import Parameter, LogParameter
from bart.parameters.planet import EccentricityParameter, CosParameter
from bart.results import Column

import sys
import h5py
import numpy as np
import matplotlib.pyplot as pl


def simple_koi(kepid="7906882"):
    # Fetch the parameters from the API.
    api = kepler.API()
    data = api.kois(kepid=kepid)
    assert data is not None and len(data) > 0, "No KOIs."

    # Download the data files.
    files = api.data(kepid).fetch_all("data")

    # Add the planets
    planets = []
    mass = 0.0
    for d in data:
        rstar = float(d.get("Stellar Radius", 1.0))
        teff = float(d["Teff"])
        logg = float(d["log(g)"])
        feh = float(d.get("koi_smet", 0.0))

        a = float(d["a/R"]) * rstar
        r = float(d["r/R"]) * rstar
        i = float(d.get("koi_incl", 90.0))
        P = float(d["Period"])
        t0 = float(d["Time of Transit Epoch"]) % P

        planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0)
        planet.parameters.append(Parameter(r"$r$", "r"))
        planet.parameters.append(LogParameter(r"$a$", "a"))
        planet.parameters.append(Parameter(r"$t_0$", "t0"))

        mass += planet.get_mstar(P) / len(data)

        planets.append(planet)

    # Set up the star.
    ldp = kepler.fiducial_ldp(teff, logg, feh, bins=40)
    star = bart.Star(mass=mass, radius=rstar, ldp=ldp)

    # Set up the system.
    system = bart.PlanetarySystem(star, iobs=i)
    [system.add_planet(p) for p in planets]
    system.parameters.append(CosParameter(r"$i$", "iobs"))

    # Add the datasets.
    for fn in files:
        system.add_dataset(KeplerDataset(fn))

    [pl.plot(d.time % P, d.flux, ".k", alpha=0.1) for d in system.datasets]
    ts = np.linspace(t0 - 1, t0 + 1, 5000)
    pl.plot(ts, system.lightcurve(ts))
    pl.xlim(t0 - 1, t0 + 1)
    pl.savefig("initial_lc.png")

    if "-r" in sys.argv:
        with h5py.File(sys.argv[sys.argv.index("-r") + 1]) as f:
            start = f["mcmc"]["chain"][:, -1, :]
        bi = [10]
    else:
        start = None
        bi = []

    if not "--results_only" in sys.argv:
        system.fit(5000, thin=10, burnin=bi, nwalkers=16, start=start)

    results = system.results(thin=1, burnin=20)
    results.lc_plot()
    results.time_plot()

    results.corner_plot([
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
            Column(r"$i$", lambda s: s.iobs),
        ])


if __name__ == "__main__":
    simple_koi()

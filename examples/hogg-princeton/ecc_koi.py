#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import bart
from bart import kepler
from bart.dataset import KeplerDataset
from bart.parameters.priors import GaussianPrior
from bart.parameters.base import Parameter, LogParameter
from bart.parameters.planet import EccentricityParameter, CosParameter
from bart.results import Column

import sys
import h5py
import numpy as np
import matplotlib.pyplot as pl


class GP2(GaussianPrior):

    def __call__(self, obj):
        v = 0.75 * obj.star.mass / np.pi / obj.star.radius ** 3
        return super(GP2, self).__call__(v)


def main():
    api = kepler.API()
    data = api.kois(kepoi="686.01")[0]
    files = api.data(data["Kepler ID"]).fetch_all("data")

    rstar = float(data["Stellar Radius"])
    teff = float(data["Teff"])
    logg = float(data["log(g)"])
    feh = 0.0

    a = float(data["a/R"]) * rstar
    r = 1.03 * float(data["r/R"]) * rstar
    e = 0.62  # from Dawson & Johnson
    pomega = 0.4 * np.pi
    i = 90.0
    P = float(data["Period"])
    t0 = float(data["Time of Transit Epoch"]) % P

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0, e=e, pomega=pomega)

    # Set up the star.
    ldp = kepler.fiducial_ldp(teff, logg, feh, bins=40, alpha=1.)
    star = bart.Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)

    # Set up the system.
    system = bart.PlanetarySystem(star, iobs=i)
    system.add_planet(planet)

    # Add the datasets.
    for fn in files:
        system.add_dataset(KeplerDataset(fn))

    [pl.plot(d.time % P, d.flux, ".k", alpha=0.1) for d in system.datasets]
    ts = np.linspace(t0 - 1, t0 + 1, 5000)
    pl.plot(ts, system.lightcurve(ts))
    pl.xlim(t0 - 1, t0 + 1)
    pl.savefig("initial_lc.png")

    # Priors.
    system.add_prior(GP2(1.02, 0.35))

    planet.parameters.append(Parameter(r"$r$", "r"))
    planet.parameters.append(LogParameter(r"$a$", "a"))
    planet.parameters.append(Parameter(r"$t_0$", "t0"))
    planet.parameters.append(EccentricityParameter())

    system.parameters.append(CosParameter(r"$i$", "iobs"))

    star.parameters.append(LogParameter(r"$R_\star$", "radius"))

    if "-r" in sys.argv:
        with h5py.File(sys.argv[sys.argv.index("-r") + 1]) as f:
            start = f["mcmc"]["chain"][:, -1, :]
        bi = [10]
    else:
        start = None
        bi = []

    if not "--results_only" in sys.argv:
        system.fit(2000, thin=10, burnin=bi, nwalkers=24, start=start)

    results = system.results(thin=1, burnin=20)
    results.lc_plot()
    results.time_plot()

    results.corner_plot([
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
            Column(r"$i$", lambda s: s.iobs),
            Column(r"$e$", lambda s: s.planets[0].e),
            Column(r"$\varpi$", lambda s: s.planets[0].pomega),
            Column(r"$\rho_\star$", lambda s: (0.75 * s.star.mass / np.pi
                                               / s.star.radius ** 3)),
        ])


if __name__ == "__main__":
    main()

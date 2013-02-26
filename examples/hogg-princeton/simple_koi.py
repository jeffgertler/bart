#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fit an arbitrary KOI using Bart.

Usage: simple_koi.py KOI [KOI...] [--results_only] [-n STEPS] [-b BURN]
                     [-s INIT]

Options:
    -h --help       show this
    KOI             the KOI number or Kepler ID to fit
    --results_only  only plot the results, don't do the fit
    -n STEPS        the number of steps to take [default: 1500]
    -b BURN         the number of burn in steps to take [default: 50]
    -s INIT         initialize from a previous run

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import bart
from bart import kepler
from bart.dataset import KeplerDataset
from bart.parameters.base import Parameter, LogParameter
from bart.parameters.planet import CosParameter
from bart.results import Column

import h5py
import numpy as np
import matplotlib.pyplot as pl


def simple_koi(kepoi=None, kepid=None, restart=None, results_only=False,
               nsteps=1500, burnin=40):
    # Fetch the parameters from the API.
    api = kepler.API()

    if kepid is None:
        assert kepoi is not None, "Choose a KOI or Kepler ID"
        # Find the Kepler ID associated with this KOI.
        data = api.kois(kepoi=kepoi)
        assert data is not None and len(data) > 0, "Unknown KOI."
        kepid = data[0]["Kepler ID"]

    data = api.kois(kepid=kepid)

    # Download the data files.
    files = api.data(kepid).fetch_all("{0}/data".format(kepid))

    # Add the planets
    planets = []
    mass = 0.0
    for d in data:
        rstar = float(d.get("Stellar Radius", 1.0))
        teff = float(d["Teff"])
        logg = float(d["log(g)"])
        feh = d["Metallicity"]
        feh = 0.0 if feh == "" else float(feh)

        a = float(d["a/R"]) * rstar
        r = float(d["r/R"]) * rstar

        # Compute the inclination using the impact parameter because the
        # cataloged inclination seems to be wrong.
        b = float(d["Impact Parameter"])
        i = np.degrees(np.arccos(b * rstar / a))
        # i = float(d["Inclination"])
        P = float(d["Period"])
        t0 = float(d["Time of Transit Epoch"]) % P

        planet = bart.Planet(r=r, a=a, t0=t0, ix=90 - i)
        planet.parameters.append(Parameter(r"$r$", "r"))
        planet.parameters.append(LogParameter(r"$a$", "a"))
        planet.parameters.append(Parameter(r"$t_0$", "t0"))
        planet.parameters.append(CosParameter(r"$i$", "ix"))

        mass += planet.get_mstar(P) / len(data)

        planets.append(planet)

    print("{0} planets".format(len(planets)))

    # Set up the star.
    ldp = kepler.fiducial_ldp(teff, logg, feh, bins=40)
    star = bart.Star(mass=mass, radius=rstar, ldp=ldp)

    # Set up the system.
    system = bart.PlanetarySystem(star, basepath="{0}".format(kepid))
    [system.add_planet(p) for p in planets]

    # Add the datasets.
    for fn in files:
        system.add_dataset(KeplerDataset(fn))
    print("{0} data points".format(sum([len(d.time)
                                                for d in system.datasets])))

    # Plot the initial fit.
    for i, p in enumerate(planets):
        P = p.get_period(star.mass)
        t0 = p.t0
        pl.clf()
        [pl.plot(d.time % P, d.flux, ".k", alpha=0.1) for d in system.datasets]
        ts = np.linspace(t0 - 1, t0 + 1, 5000)
        pl.plot(ts, system.lightcurve(ts))
        pl.xlim(t0 - 1, t0 + 1)
        pl.savefig("{0}/initial_lc_{1}.png".format(kepid, i))

    if restart is not None:
        with h5py.File(restart) as f:
            start = f["mcmc"]["chain"][:, -1, :]
        bi = [10]
    else:
        start = None
        bi = []

    if not results_only:
        system.fit(nsteps, thin=10, burnin=bi,
                   nwalkers=16, start=start)

    results = system.results(thin=1, burnin=burnin)
    results.lc_plot()
    results.time_plot()

    results.corner_plot([
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
            Column(r"$i$", lambda s: 90 - s.planets[0].ix),
        ])


if __name__ == "__main__":
    from docopt import docopt
    args = docopt(__doc__)
    for koi in args["KOI"]:
        print("Starting KOI {0}".format(koi))
        try:
            kwargs = {}
            if "." in koi:
                kwargs["kepoi"] = koi
            else:
                kwargs["kepid"] = koi
            simple_koi(restart=args["-s"],
                       results_only=args["--results_only"],
                       nsteps=int(args["-n"]), burnin=int(args["-b"]),
                       **kwargs)
        except:
            print("KOI {0} failed.".format(koi))
            import traceback
            traceback.print_exc()

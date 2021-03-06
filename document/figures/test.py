#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo of how you would use Bart to fit a Kepler light curve.

Usage: kepler6.py [FILE...] [-e ETA]... [--results_only] [-n STEPS] [-b BURN]
                  [--rv]

Options:
    -h --help       show this
    FILE            a list of FITS files including the data
    -e ETA          the strength of the LDP prior [default: 0.05]
    --results_only  only plot the results, don't do the fit
    -n STEPS        the number of steps to take [default: 2000]
    -b BURN         the number of burn in steps to take [default: 50]
    --rv            fit radial velocity?

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))))
import bart
from bart import kepler
from bart.dataset import KeplerDataset, RVDataset
from bart.results import ResultsProcess, Column
from bart.parameters.priors import GaussianPrior
from bart.parameters.base import Parameter, LogParameter
from bart.parameters.star import RelativeLimbDarkeningParameters
from bart.parameters.planet import EccentricityParameter

import numpy as np
import matplotlib.pyplot as pl
from matplotlib.ticker import MaxNLocator


np.random.seed(100)


class CosParameter(Parameter):

    def getter(self, obj):
        return np.cos(np.radians(obj.iobs))

    def setter(self, obj, val):
        obj.iobs = np.degrees(np.arccos(val))

    def sample(self, obj, std=1e-5, size=1):
        return np.cos(np.radians(obj.iobs * (1 + std * np.random.randn(size))))


def main(fns, eta, results_only=False, nsteps=2000, nburn=50, fitrv=True):
    # Initial physical parameters from:
    #  http://kepler.nasa.gov/Mission/discoveries/kepler6b/
    #  http://arxiv.org/abs/1001.0333

    rstar = 1.391  # +0.017 -0.034 R_sun
    Teff = 5647.
    logg = 4.24
    feh = 0.34

    P = 3.234723  # ± 0.000017 days
    a = 7.05  # +0.11 -0.06 R_*
    r = 0.09829  # +0.00014 -0.00050 R_*
    i = 86.8  # ± 0.3 degrees
    mass = 0.669 * 9.5492e-4  # Solar masses.

    # The reference "transit" time.
    t0 = 1.795  # 0.28  # Found by eye.

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0, mass=mass)

    # Set up the star.
    ldp = kepler.fiducial_ldp(Teff, logg, feh, bins=15, alpha=0.5)
    star = bart.Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)

    # Set up the system.
    #system = bart.PlanetarySystem(star, iobs=i, rv0=-15.0,
    #                              basepath="kepler6-{0}".format(eta))
    system = bart.PlanetarySystem(star, iobs=i)
    system.add_planet(planet)

    # Fit parameters.
    planet.parameters.append(Parameter(r"$r$", "r"))
    planet.parameters.append(LogParameter(r"$a$", "a"))
    planet.parameters.append(Parameter(r"$t_0$", "t0"))
    system.parameters.append(CosParameter(r"$i$", "iobs"))
    
    # Read in the data.
    for fn in fns:
        system.add_dataset(KeplerDataset(fn))
        print(system.datasets[-1].time)

    # Add the RV data.
    rv = np.loadtxt("k6-rv.txt")
    ds = RVDataset(rv[:, 0], rv[:, 2], rv[:, 3], jitter=5.0)
    print(ds.time)
    if fitrv:
        ds.parameters.append(LogParameter(r"$\delta_v$", "jitter"))
        system.add_dataset(ds)
    
    print(planet.get_period(star.mass))

    # Plot initial conditions.
    pl.plot(system.datasets[0].time % P, system.datasets[0].flux, ".k",
            alpha=0.1)
    ts = np.linspace(0, P, 5000)
    pl.plot(ts, system.lightcurve(ts))
    pl.savefig("initial_lc.png")
    pl.clf()
    pl.errorbar(ds.time % P, ds.rv,
                yerr=np.sqrt(ds.rverr ** 2 + ds.jitter ** 2), fmt=".k")
    pl.plot(ts, system.radial_velocity(ts))
    pl.savefig("initial_rv.png")
    
    t = np.linspace(-0.2, 0.2, 5000)
    pl.clf()
    pl.plot(t, system.lightcurve(t))
    pl.savefig("lightcurve.png")

    print(system.vector)
    if not results_only:
        system.fit(nsteps, thin=10, burnin=[], nwalkers=64) 
    
    results = system.results(thin=10, burnin=nburn)
    
'''    results.corner_plot([
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
            Column(r"$i$", lambda s: s.iobs),
        ])
'''


def download_data(bp):
    # Get the data.
    api = kepler.API()
    print("Downloading the data files.")
    return api.data("10874614").fetch_all(basepath=bp)


def quantiles(r):
    r = np.sort(r)
    l = len(r)
    return [r[int(q * l)] for q in [0.16, 0.50, 0.84]]


if __name__ == "__main__":
    # Parse the command line arguments.
    from docopt import docopt
    args = docopt(__doc__)

    in_fns = args["FILE"]

    # Download the data files.
    if len(in_fns) > 0:
        bp = os.path.dirname(in_fns[0])
    else:
        bp = "kepler6/data"
    fns = download_data(bp)
    print("  .. Finished.")

    # Figure out which data files to use.
    if len(in_fns) > 0:
        assert all([fn in fns for fn in in_fns])
    else:
        in_fns = fns

    # Run the fit.
    etas = np.array([float(eta) for eta in args["-e"]])
    for eta in etas:
        main(in_fns, eta, results_only=args["--results_only"],
             nsteps=int(args["-n"]), nburn=int(args["-b"]),
             fitrv=args["--rv"])


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


# Reproducible Science.™
np.random.seed(100)


class CosParameter(Parameter):

    def getter(self, obj):
        return np.cos(np.radians(obj.iobs))

    def setter(self, obj, val):
        obj.iobs = np.degrees(np.arccos(val))

    def sample(self, obj, std=1e-5, size=1):
        return np.cos(np.radians(obj.iobs * (1 + std * np.random.randn(size))))


def main(fns, eta, results_only=False, nsteps=50, nburn=50, fitrv=True):
    # Initial physical parameters from:
    #  http://kepler.nasa.gov/Mission/discoveries/kepler6b/
    #  http://arxiv.org/abs/1001.0333

    rstar = 1.391  # +0.017 -0.034 R_sun
    Teff = 5647.
    logg = 4.24
    feh = 0.34

    P = 3.234723  # ± 0.000017 days
    r = 0.09829  # +0.00014 -0.00050 R_*
    i = 86.8  # ± 0.3 degrees
    mass = 0.669 * 9.5492e-4  # Solar masses.


    ldp = kepler.fiducial_ldp(teff=5647, logg=4.236, feh=0.34, bins=50)
    star = bart.Star(mass=.602, radius=.01234, ldp=ldp, flux = 100.)
    #star = bart.Star(mass=.00005, radius=1.391, ldp=ldp, flux = 100.)

    a = 7.05  # +0.11 -0.06 R_*
    Rp = .7 * star.radius
    planet = bart.Planet(a=100, r=.01234, t0=0, mass=0.0009551)

    kepler6 = bart.PlanetarySystem(star, iobs=86.8)
    kepler6.add_planet(planet)

    tbin = .01
    t = np.arange(-100, 100, tbin)
    times = kepler6.photons(t, tbin = tbin)
    
    
    planet.parameters.append(LogParameter(r"$r$", "r"))
    '''
    planet.parameters.append(Parameter(r"$t_0$", "t0"))
    planet.parameters.append(LogParameter(r"$a$", "a"))
    kepler6.parameters.append(CosParameter(r"$i$", "iobs"))
    '''
    
    print(kepler6.vector)
    if not results_only:
        kepler6.fit(nsteps, thin=10, burnin=[], nwalkers=64) 
    
    results = kepler6.results(thin=10, burnin=nburn)
    
    print(len(kepler6.liketrace))
    print(np.max(kepler6.liketrace))
    print(np.min(kepler6.liketrace))
    pl.clf()
    pl.plot(range(len(kepler6.liketrace)), kepler6.liketrace)
    pl.savefig("likelihood_trace")


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


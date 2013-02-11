#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo of how you would use Bart to fit a Kepler light curve.

Usage: kepler6.py [FILE...] [-e ETA]...

Options:
    -h --help  show this.
    -e ETA     list of the strengths of the LDP prior. [default: 0.05]
    FILE       a list of FITS files including the data.

"""

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                   os.path.abspath(__file__)))))
import bart
from bart import kepler
from bart.dataset import KeplerDataset
from bart.results import Column
from bart.parameters.base import Parameter, LogParameter
from bart.parameters.star import RelativeLimbDarkeningParameters

import numpy as np


# Reproducible Science.™
np.random.seed(100)


class CosParameter(Parameter):

    def getter(self, obj):
        return np.cos(np.radians(obj.iobs))

    def setter(self, obj, val):
        obj.iobs = np.degrees(np.arccos(val))

    def sample(self, obj, std=1e-5, size=1):
        return np.cos(np.radians(obj.iobs * (1 + std * np.random.randn(size))))


def main(fns, eta):
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

    # The reference "transit" time.
    t0 = 1.795  # Found by eye.

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0)

    # Set up the star.
    ldp = kepler.fiducial_ldp(Teff, logg, feh, bins=15, alpha=0.5)
    star = bart.Star(mass=planet.get_mstar(P), radius=rstar, ldp=ldp)

    # Set up the system.
    system = bart.PlanetarySystem(star, iobs=i,
                                  basepath="kepler6-{0}".format(eta))
    system.add_planet(planet)

    # Fit parameters.
    planet.parameters.append(Parameter(r"$r$", "r"))
    planet.parameters.append(LogParameter(r"$a$", "a"))
    planet.parameters.append(Parameter(r"$t_0$", "t0"))

    system.parameters.append(CosParameter(r"$i$", "iobs"))

    star.parameters.append(RelativeLimbDarkeningParameters(star.ldp.bins,
                                                   star.ldp.intensity,
                                                   eta=eta))

    # Read in the data.
    for fn in fns:
        system.add_dataset(KeplerDataset(fn))

    # system.fit((time, flux, ferr), 1, thin=1, burnin=[], nwalkers=64)
    system.fit(2000, thin=10, burnin=[], nwalkers=64)

    # Plot the results.
    print("Plotting results")
    results = system.results(thin=10, burnin=50)
    results.latex([
            Column(r"$P\,[\mathrm{days}]$",
                   lambda s: s.planets[0].get_period(s.star.mass)),
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0\,[\mathrm{days}]$", lambda s: s.planets[0].t0),
            Column(r"$i\,[\mathrm{deg}]$", lambda s: s.iobs),
        ])

    results.lc_plot()
    results.ldp_plot(fiducial=kepler.fiducial_ldp(Teff, logg, feh))
    results.time_plot()
    results.corner_plot([
            Column(r"$a$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0$", lambda s: s.planets[0].t0),
            Column(r"$i$", lambda s: s.iobs),
        ])


def download_data(bp):
    # Get the data.
    api = kepler.API()
    print("Downloading the data files.")
    return api.data("10874614").fetch_all(basepath=bp)


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
    for eta in args["-e"]:
        main(fns, float(eta))

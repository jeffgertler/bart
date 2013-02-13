#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Demo of how you would use Bart to fit a Kepler light curve.

Usage: kepler6.py [FILE...] [-e ETA]... [--results_only] [-n STEPS] [-b BURN]

Options:
    -h --help       show this
    FILE            a list of FITS files including the data
    -e ETA          the strength of the LDP prior [default: 0.05]
    --results_only  only plot the results, don't do the fit
    -n STEPS        the number of steps to take [default: 2000]
    -b BURN         the number of burn in steps to take [default: 50]

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
from bart.results import ResultsProcess, Column
from bart.parameters.base import Parameter, LogParameter
from bart.parameters.star import RelativeLimbDarkeningParameters

import numpy as np
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


def main(fns, eta, results_only=False, nsteps=2000, nburn=50):
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
    t0 = 1.795  # Found by eye.

    # Set up the planet.
    planet = bart.Planet(r=r * rstar, a=a * rstar, t0=t0, mass=mass)

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
    if not results_only:
        system.fit(nsteps, thin=10, burnin=[], nwalkers=64)

    # Plot the results.
    print("Plotting results")
    results = system.results(thin=10, burnin=nburn)
    results.latex([
            Column(r"$P\,[\mathrm{days}]$",
                   lambda s: s.planets[0].get_period(s.star.mass)),
            Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
            Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
            Column(r"$t_0\,[\mathrm{days}]$", lambda s: s.planets[0].t0),
            Column(r"$i\,[\mathrm{deg}]$", lambda s: s.iobs),
        ])

    # RV plot.
    rv = np.loadtxt("k6-rv.txt")
    ax = results._rv_plots("rv")[0]
    ax.errorbar(24 * (rv[:, 0] % P), rv[:, 2], yerr=rv[:, 3], fmt=".k")
    ax.figure.savefig("rv.png")

    # Other results plots.
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
             nsteps=int(args["-n"]), nburn=int(args["-b"]))

    # Plot the combined histogram.
    import matplotlib.pyplot as pl
    # hist_ax = pl.figure(figsize=(4, 4)).add_axes([0.1, 0.2, 0.8, 0.75])

    columns = {
            "r": Column(r"$r / R_\star$",
                        lambda s: float(s.planets[0].r / s.star.radius)),
            "a": Column(r"$a / R_\star$",
                        lambda s: float(s.planets[0].a / s.star.radius)),
            "i": Column(r"$i$", lambda s: float(s.iobs)),
        }
    axes = dict([(k, pl.figure(figsize=(4, 3.5))
                       .add_axes([0.25, 0.2, 0.7, 0.7])) for k in columns])

    for eta in np.sort(etas)[::-1]:
        results = ResultsProcess(basepath="kepler6-{0}".format(eta),
                                 burnin=50)

        # Get the samples.
        samples = dict([(k, []) for k in columns])
        for s in results.itersteps():
            for k, c in columns.iteritems():
                samples[k].append(c.getter(s))

        # Compute quantiles.
        for k, s in samples.iteritems():
            mn, md, mx = quantiles(s)
            axes[k].plot(eta, md, "ok")
            axes[k].plot([eta, eta], [mn, mx], "k")

    mn, mx = etas.min(), etas.max()
    d = mx - mn
    for k, ax in axes.iteritems():
        ax.set_xlim([0, mx + 0.1 * d])
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        [l.set_rotation(45) for l in ax.get_yticklabels()]
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(columns[k].name)
        ax.figure.savefig("k6-{0}.png".format(k))
        ax.figure.savefig("k6-{0}.pdf".format(k))

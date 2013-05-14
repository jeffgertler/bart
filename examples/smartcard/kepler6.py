#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
sys.path.insert(0,
                os.path.join(os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.abspath(__file__))))))


import kplr
import numpy as np
import matplotlib.pyplot as pl

import bart
from bart.results import Column
from bart.dataset import LCDataset
from bart.ldp import QuadraticLimbDarkening
from bart.parameters.base import Parameter, LogParameter, CosParameter
from bart.parameters.priors import UniformPrior


def fit_single(pnm):
    client = kplr.API()
    koi = client.planet(pnm).koi

    # Stellar parameters.
    rstar = koi.koi_srad
    teff = koi.koi_steff
    logg = koi.koi_slogg
    feh = koi.koi_smet

    # Planet parameters.
    period = koi.koi_period
    a = koi.koi_dor * rstar
    r = koi.koi_ror * rstar
    i = koi.koi_incl
    t0 = koi.koi_time0bk % period

    # Limb darkening coefficients.
    mu1, mu2 = kplr.ld.get_quad_coeffs(teff, logg=logg, feh=feh,
                                       model="claret11")
    ldp = QuadraticLimbDarkening(mu1, mu2).histogram(
        np.linspace(0, 1, 20)[1:])

    # Set up system.
    planet = bart.Planet(r=r, a=a, t0=t0)
    star = bart.Star(mass=planet.get_mstar(period), radius=rstar, ldp=ldp)
    system = bart.PlanetarySystem(star, iobs=i)
    system.add_planet(planet)

    # Load the datasets.
    ax1 = pl.figure().add_subplot(111)
    ax2 = pl.figure().add_subplot(111)
    for d in koi.data:
        if "slc" in d.filename:
            continue
        print(d.filename)
        d.fetch()
        dataset = kplr.Dataset(d.filename, untrend=True)
        ds = LCDataset(dataset.time, dataset.flux, dataset.ferr,
                       dataset.texp)
        ds.cadence = 1 if "llc" in d.filename else 0
        system.add_dataset(ds)
        ax1.plot(dataset.time, dataset.flux, ".k", ms=0.5)
        ax2.plot(dataset.time % period, dataset.flux, ".k", alpha=0.1)

    # Save the plot of the unfolded data.
    ax1.set_ylim(0.988, 1.0015)
    ax1.set_xlim(100, 1400)
    ax1.set_xlabel("time [KBJD]")
    ax1.figure.savefig("unfolded.pdf")

    # Plot the initial light curve.
    t = np.linspace(0, period, 500)
    lc = system.lightcurve(t)
    ax2.plot(t, lc, "r")
    ax2.set_xlim(0, period)
    ax2.figure.savefig("initial.png")

    # Decide which parameters should be fit for.
    planet.parameters.append(Parameter(r"$r$", "r", prior=UniformPrior(0, 1)))
    planet.parameters.append(LogParameter(r"$a$", "a",
                                          prior=UniformPrior(1, 20)))
    planet.parameters.append(Parameter(r"$t_0$", "t0",
                                       prior=UniformPrior(0, 10)))
    system.parameters.append(CosParameter(r"$i$", "iobs",
                                          prior=UniformPrior(85, 90)))

    # Run MCMC.
    system.run_mcmc(2000, thin=10, nwalkers=16)
    results = system.results(burnin=50)
    results.lc_plot()
    results.time_plot()
    results.corner_plot([
        Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
        Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
        Column(r"$t_0$", lambda s: s.planets[0].t0),
        Column(r"$i$", lambda s: s.iobs),
    ])


if __name__ == "__main__":
    fit_single("6b")

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


client = kplr.API()


def get_kois():
    koi = client.koi("571.01")
    kepid = koi.kepid
    return sorted(list(client.kois(kepid=kepid)), key=lambda o: o.koi_period)


def fit(nplanets=4):
    kois = get_kois()[:nplanets]

    # Stellar parameters.
    koi = kois[0]
    teff = koi.koi_steff
    logg = koi.koi_slogg
    feh = koi.koi_smet

    # Limb darkening coefficients.
    mu1, mu2 = kplr.ld.get_quad_coeffs(teff, logg=logg, feh=feh,
                                       model="claret11")
    ldp = QuadraticLimbDarkening(mu1, mu2).histogram(
        np.sqrt(np.linspace(0, 1, 100)[1:]))

    # Set up the star and the system.
    star = bart.Star(ldp=ldp)
    system = bart.PlanetarySystem(star, iobs=90.0)

    # Loop over planets.
    masses, periods = [], []
    for koi in kois:
        period = koi.koi_period
        a = koi.koi_dor
        r = koi.koi_ror
        i = koi.koi_incl
        t0 = koi.koi_time0bk % period
        planet = bart.Planet(r=r, a=a, t0=t0, ix=90 - i)
        system.add_planet(planet)
        masses.append(planet.get_mstar(period))
        periods.append(period)

    # Set the stellar mass.
    star.mass = np.mean(masses)

    # Update all the semi-majors.
    for planet, period in zip(system.planets, periods):
        planet.a = star.get_semimajor(period)

    # Load the datasets.
    ts, fs = [], []
    for d in koi.data:
        if "slc" in d.filename:
            continue
        print(d.filename)
        d.fetch()
        dataset = kplr.Dataset(d.filename, untrend=True, dt=1.5)
        print(np.sum(dataset.quality[dataset.mask]))
        ds = LCDataset(dataset.time[dataset.mask], dataset.flux[dataset.mask],
                       dataset.ferr[dataset.mask], dataset.texp)
        ds.cadence = 1 if "llc" in d.filename else 0
        system.add_dataset(ds)
        ts.append(dataset.time[dataset.mask])
        fs.append(dataset.flux[dataset.mask])
    ts = np.concatenate(ts)
    fs = np.concatenate(fs)

    pl.plot(ts, fs, ".k", ms=2)
    pl.xlim(ts.min(), ts.max())
    pl.savefig("unfolded.png")
    pl.xlim(1100, 1150)
    pl.savefig("unfolded-zoom.png")

    for i, planet in enumerate(system.planets):
        period = planet.get_period(star.mass)
        duration = period / np.pi / planet.a
        t0 = planet.t0

        # Compute light curve.
        t = np.linspace(t0 - duration, t0 + duration, 5000)
        lc = system.lightcurve(t)

        # Plot the initial light curve.
        pl.clf()
        pl.plot((ts - t0 + 0.5 * period) % period - 0.5 * period, fs, ".k",
                ms=1)
        pl.plot(t - t0, lc, "r")
        pl.xlim(-duration, duration)
        pl.ylim(0.998, 1.002)
        pl.savefig("initial.{0}.png".format(i))

        # Set the fit parameters.
        planet.parameters.append(Parameter(r"$r$", "r",
                                           prior=UniformPrior(0.02, 1)))
        planet.parameters.append(Parameter(r"$a$", "a",
            prior=UniformPrior(planet.a - 0.01, planet.a + 0.01)))
        planet.parameters.append(Parameter(r"$t_0$", "t0",
            prior=UniformPrior(t0 - 0.0001, t0 + 0.0001)))
        # planet.parameters.append(LogParameter(r"$\delta i_x$", "ix",
        #                                    prior=UniformPrior(0, 5)))

    # Add star parameters.
    print(star.mass)
    star.parameters.append(LogParameter(r"$M_\star$", "mass",
                                        prior=UniformPrior(0, 10)))

    # Run MCMC.
    # system.run_mcmc(200, thin=10, nwalkers=16)
    results = system.results(burnin=0)
    results.lc_plot()
    results.time_plot()
    results.corner_plot([
        Column(r"$a/R_\star$", lambda s: s.planets[0].a / s.star.radius),
        Column(r"$r/R_\star$", lambda s: s.planets[0].r / s.star.radius),
        Column(r"$t_0$", lambda s: s.planets[0].t0),
        Column(r"$M_\star$", lambda s: s.star.mass),
        Column(r"$\ln p$", lambda s: s.lnprob()),
    ])


if __name__ == "__main__":
    fit(1)

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import sys
import emcee
import time as timer
import triangle
import matplotlib.pyplot as pl
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))

import bart
from bart.parameters import (Parameter, ImpactParameter, PeriodParameter,
                             LogMultiParameter, LogParameter)
from bart.priors import UniformPrior, NormalPrior


def setup():
    bins = np.linspace(0, 1, 50)[1:] ** 0.5
    ldp = bart.ld.QuadraticLimbDarkening(0.3, 0.1).histogram(bins)

    # Set up the planetary system.
    star = bart.Star(ldp=ldp, flux=5000.0)
    planet = bart.Planet(1.1, 400., t0=2.5)
    ps = bart.PlanetarySystem(star)
    ps.add_planet(planet)

    # Initialize the model.
    model = bart.Model(ps)

    # Generate some fake data.
    dt = 0.001
    tbins = np.arange(0.0, 5.0, dt)
    lc = ps.lightcurve(tbins, texp=0, K=1)
    times = np.array([])
    for t, r in zip(tbins, lc):
        times = np.append(times, t + dt *
                          np.random.rand(np.random.poisson(lam=dt * r)))

    n, bins, p = pl.hist(times, 50)
    print(bins[1] - bins[0])
    pl.plot(tbins, lc)
    pl.savefig("data.png")

    # Add the dataset.
    dataset = bart.data.PhotonStream(times)
    model.datasets.append(dataset)

    # Add some parameters.
    model.parameters.append(Parameter(star, "flux"))
    model.parameters.append(LogParameter(planet, "r"))

    return model


if __name__ == "__main__":
    model = setup()

    # Set up sampler.
    nwalkers = 20
    v = model.vector
    p0 = v[None, :] * (1e-4 * np.random.randn(len(v) * nwalkers)
                       + 1).reshape((nwalkers, len(v)))
    sampler = emcee.EnsembleSampler(nwalkers, len(p0[0]), model,
                                    threads=nwalkers)

    # Run a burn-in.
    pos, lnprob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()

    fn = "samples.txt"
    with open(fn, "w") as f:
        f.write("# {0}\n".format(" ".join(map(unicode, model.parameters))))

    strt = timer.time()
    for pos, lnprob, state in sampler.sample(pos, lnprob0=lnprob,
                                             iterations=2000,
                                             storechain=False):
        with open(fn, "a") as f:
            for p, lp in zip(pos, lnprob):
                f.write("{0} {1}\n".format(
                    " ".join(map("{0}".format, p)), lp))

    print("Took {0} seconds".format(timer.time() - strt))
    print("Acceptance fraction: {0}"
          .format(np.mean(sampler.acceptance_fraction)))

    samples = np.loadtxt(fn)
    figure = triangle.corner(samples)
    figure.savefig("triangle.png")

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
from bart.parameters import (Parameter, LogParameter, ImpactParameter)
from bart.priors import (UniformPrior, NormalPrior)

ITERATIONS = 2000


def setup():
    bins = np.linspace(0, 1, 50)[1:] ** 0.5
    ldp = bart.ld.QuadraticLimbDarkening(0.3, 0.1).histogram(bins)

    # Set up the planetary system.
    star = bart.Star(mass=.602, radius=.01234, ldp=ldp, flux=500.0)
    planet = bart.Planet(0.1005, 4332.59, mass=0.0009551, t0=2.5)
    #planet = bart.Planet(0.01005, 44332.59, mass=0.0009551, t0=2.5)
    ps = bart.PlanetarySystem(star)
    ps.add_planet(planet)

    # Initialize the model.
    model = bart.Model(ps)

    # Generate some fake data.
    dt, bglevel = 0.001, 10.0
    tbins = np.arange(0.0, 5.0, dt)
    lc = ps.lightcurve(tbins, texp=0, K=1) + bglevel
    times = np.array([])
    for t, r in zip(tbins, lc):
        times = np.append(times, t + dt *
                          np.random.rand(np.random.poisson(lam=dt * r)))

    n, bins, p = pl.hist(times, 50)
    pl.plot(tbins, lc * (bins[1] - bins[0]))
    pl.savefig("data.png")

    # Add the dataset.
    dataset = bart.data.PhotonStream(times, background=bglevel)
    model.datasets.append(dataset)
    
    # Create priors
    fluxPrior = bart.priors.UniformPrior(0, star.flux*2)
    #planetRPrior = bart.priors.UniformPrior(0, planet.r * 2)
    
    # Add some parameters.
    model.parameters.append(Parameter(star, "flux", lnprior = fluxPrior))
    model.parameters.append(LogParameter(star, "mass"))
    model.parameters.append(LogParameter(star, "radius"))
    model.parameters.append(Parameter(planet, "mass"))
    model.parameters.append(Parameter(planet, "r"))
    model.parameters.append(Parameter(planet, "t0"))
    model.parameters.append(ImpactParameter(planet))
    

    return model


def print_trace(file_name, index, samples):
    print("plotting " + file_name)
    pl.clf()
    walkers = np.arange(nwalkers)
    step = np.arange(ITERATIONS)
    test = np.empty(ITERATIONS)
    for i in walkers:
        for j in step:
            test[j] = samples[j*nwalkers+i][index]
        pl.plot(step, test)
    pl.savefig(file_name + ".png")

def print_burntrace(nwalers, nburnsteps, nburn, plot_name, index, burnsamples, burnprobs):
    print("plotting " + plot_name)
    pl.figure(figsize=[6*nburn,10])
    walkers = np.arange(nwalkers)
    step = np.arange(nburnsteps)
    test = np.empty(nburnsteps)
    for n in range(nburn):
        pl.subplot(2, nburn, n+1)
        for i in walkers:
            for j in step:
                test[j] = burnsamples[n][j+i*nburnsteps][index]
            pl.plot(step, test)
        pl.ylabel(plot_name)
        pl.subplot(2, nburn, n+nburn+1)
        pl.xlabel("steps")
        pl.ylabel("ln prob")
        for i in np.arange(nwalkers):
            for j in np.arange(nburnsteps):
                test[j] = burnprobs[n][j+i*nburnsteps]
            pl.plot(step, test)
    pl.savefig(plot_name + "_burntrace.png")


if __name__ == "__main__":
    model = setup()

    # Set up sampler.
    nwalkers = 20
    nburnsteps = 1000
    nburn = 5
    tiny = 1e-3
    v = model.vector
    p0 = v[None, :] * (tiny * np.random.randn(len(v) * nwalkers)
                       + 1).reshape((nwalkers, len(v)))
    sampler = emcee.EnsembleSampler(nwalkers, len(p0[0]), model,
                                threads=nwalkers)
    burnsamples = []
    burnprobs = []
    # Run a burn-in.
    for n in range(nburn):
        print(str(n+1) + " of " + str(nburn))
        pos, lnprob, state = sampler.run_mcmc(p0, nburnsteps)
        pbest = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
        p0 = pbest[None, :] * (tiny * np.random.randn(len(v) * nwalkers)
                     + 1).reshape((nwalkers, len(v)))
        
        burnsamples.append(sampler.flatchain)
        burnprobs.append(sampler.flatlnprobability)
        sampler.reset()

    # Printing burntrace
    print_burntrace(nwalkers, nburnsteps, nburn, "flux", 0, burnsamples, burnprobs)
    print_burntrace(nwalkers, nburnsteps, nburn, "star mass", 1, burnsamples, burnprobs)
    print_burntrace(nwalkers, nburnsteps, nburn, "star radius", 2, burnsamples, burnprobs)
    print_burntrace(nwalkers, nburnsteps, nburn, "planet mass", 3, burnsamples, burnprobs)
    print_burntrace(nwalkers, nburnsteps, nburn, "planet radius", 4, burnsamples, burnprobs)
    print_burntrace(nwalkers, nburnsteps, nburn, "t0", 5, burnsamples, burnprobs)
    print_burntrace(nwalkers, nburnsteps, nburn, "observation angle", 6, burnsamples, burnprobs)

    fn = "samples.txt"
    with open(fn, "w") as f:
        f.write("# {0} {1}\n".format(" ".join(map(unicode, model.parameters)),
                                     "ln\,p"))
    
    ln_prob=np.empty(0)
    strt = timer.time()
    for pos, lnprob, state in sampler.sample(pos, lnprob0=lnprob,
                                             iterations=ITERATIONS,
                                             storechain=False):
        ln_prob = np.append(ln_prob, lnprob)
        with open(fn, "a") as f:
            for p, lp in zip(pos, lnprob):
                f.write("{0} {1}\n".format(
                    " ".join(map("{0}".format, p)), lp))

    print("Took {0} seconds".format(timer.time() - strt))
    print("Acceptance fraction: {0}"
          .format(np.mean(sampler.acceptance_fraction)))

    samples = np.loadtxt(fn)

    print_trace("trace_planetradius", 4, samples)

    ln_prob = np.reshape(ln_prob, (1, nwalkers*ITERATIONS))
    samples = np.concatenate((samples, ln_prob.T), axis=1)
    figure = triangle.corner(samples, labels=[r"$flux$", r"log star mass", r"log star radius", r"planet mass", r"planet radius", r"t0", r"angle", r"ln prob", r""], truths = [500, -0.2204, -1.9086, 0.0009551, 0.1005, 2.5, 0, 0, 0])
    #figure = triangle.corner(samples, labels=[r"$flux$", r"log star mass", r"log star radius", r"planet mass", r"planet radius", r"t0", r"angle", r"ln prob"], truths = [500, -0.2204, -1.9086, 0.0009551, 0.1005, 2.5, 0, 0])
    figure.savefig("triangle.png")

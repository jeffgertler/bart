#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", ".."))

from bart.injection import kepler_injection
from bart import _george
import emcee
import triangle
import numpy as np
import matplotlib.pyplot as pl

# Get some datasets but don't actually
datasets, ps = kepler_injection(2301306, 400.0, 0.0)

dt = 10.0

ds = datasets[6]
tmn, tmx = ds.time.min(), ds.time.max()
t0 = tmn + (tmx - tmn) * np.random.rand()
m = (ds.time > t0) * (ds.time < t0 + dt)

time, flux, ferr = ds.time[m], ds.flux[m] - 1, ds.ferr[m]

pl.plot(time, flux, ".k")
pl.savefig("data.png")


def lnprobfn(p):
    p = np.exp(p)
    return _george.lnlikelihood(time, flux, ferr, p[0], p[1])


ndim, nwalkers = 2, 50
p0 = [np.array([np.log(1000), np.log(3)]) + 1e-5 * np.random.randn(2)
      for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn, threads=10)
sampler.run_mcmc(p0, 100)

pl.clf()
figure = triangle.corner(sampler.flatchain)
figure.savefig("triangle.png")

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
import scipy.optimize as op

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


def loss(p):
    p = np.exp(p)
    return -_george.lnlikelihood(time, flux, ferr, p[0], p[1])

p0 = [-10.0, 1.5]
results = op.minimize(loss, p0, jac=True)
print results
print results.x
print results.message

assert 0

ndim, nwalkers = 2, 10
p0 = [np.array([-10.0, 1.5]) + 1e-5 * np.random.randn(2)
      for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobfn, threads=10)
sampler.run_mcmc(p0, 1000)

# Plot predictions.
for sample in sampler.flatchain[100 * nwalkers::2 * nwalkers + 1]:
    p = np.exp(sample)
    mu, cov = _george.predict(time, flux, ferr, p[0], p[1], time)
    model = np.random.multivariate_normal(mu, cov)
    pl.plot(time, model.T, "k", alpha=0.05)
pl.savefig("prediction.png")

pl.clf()
figure = triangle.corner(sampler.flatchain)
figure.savefig("triangle.png")

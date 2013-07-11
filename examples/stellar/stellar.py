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

time, flux, ferr = None, None, None
dt = 10.0


def random_dataset():
    dsn = np.random.randint(len(datasets))
    ds = datasets[dsn]
    tmn, tmx = ds.time.min(), ds.time.max()
    t0 = tmn + (tmx - tmn) * np.random.rand()
    m = (ds.time > t0) * (ds.time < t0 + dt)
    return dsn, ds.time[m], ds.flux[m] - 1, ds.ferr[m]

data = [random_dataset()[1:] for i in range(10)]


def lnprobfn(p):
    p = np.exp(p)
    return _george.lnlikelihood(time, flux, ferr, p[0], p[1])


def full_lnprobfn(p):
    p = np.exp(p)
    return np.sum([_george.lnlikelihood(d[0], d[1], d[2], p[0], p[1])
                   for d in data])


def loss(p):
    ll, g = _george.gradlnlikelihood(time, flux, ferr, p[0], p[1])
    if np.isfinite(ll):
        return -ll, -g
    return np.inf, -g


def full_loss(p):
    ll, g = zip(*[_george.gradlnlikelihood(d[0], d[1], d[2], p[0], p[1])
                  for d in data])
    if np.all(np.isfinite(ll)):
        return -np.sum(ll), -sum(g)
    return np.inf, -g[0]


# Fit a bunch of datasets simultaneously.
p0 = [10 ** -5, 1.5]
results = op.minimize(full_loss, p0, jac=True, method="L-BFGS-B",
                      bounds=[(0, None), (0, None)])
p = results.x
print(p, results.success)

# Sample the posterior.
ndim, nwalkers = 2, 20
p0 = [np.log(p) + 1e-4 * np.random.randn(2) for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, full_lnprobfn, threads=10)
print("sampling")
sampler.run_mcmc(p0, 1000)

print("plotting")
pl.clf()
figure = triangle.corner(sampler.flatchain, truths=np.log(p))
figure.savefig("triangle.png")

assert 0


fn = "hyperpars.txt"
for i in range(200):
    dsn, time, flux, ferr = random_dataset()

    p0 = [10 ** -5, 0.1]
    results = op.minimize(loss, p0, jac=True, method="L-BFGS-B",
                          bounds=[(0, None), (0, None)])

    p = results.x
    print(dsn, p, results.success)
    if results.success:
        with open(fn, "a") as f:
            f.write("{0} {1} {2}\n".format(dsn, p[0], p[1]))

assert 0

# Plot predictions.
t = np.linspace(time.min(), time.max(), 100)
mu, cov = _george.predict(time, flux, ferr, p[0], p[1], t)
model = np.random.multivariate_normal(mu, cov, size=50)
pl.plot(t, model.T, "k", alpha=0.05)
pl.savefig("prediction.png")

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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["estimate_gp_hyperpars"]


import numpy as np
from multiprocessing import Pool

from . import _george

try:
    import scipy.optimize as op
    op = op
except ImportError:
    op = None


def estimate_gp_hyperpars(datasets, dt=10.0, nblocks=20):
    """
    Estimate the hyper-parameters of a Gaussian process noise model for a
    set of :class:`data.GPLightCurve` datasets. The hyper-parameters will be
    updated in place.

    :param datasets:
        A list of :class:`GPLightCurve` datasets.

    :param dt:
        The length of blocks to use in days. (default: 10.0)

    :param nblocks:
        The number of blocks to fit. (default: 20)

    """
    if op is None:
        raise ImportError("You need to install scipy for this function.")

    # Choose nblocks blocks of data from the datasets.
    data = []
    for n in range(nblocks):
        # Choose a random dataset.
        ds = datasets[np.random.randint(len(datasets))]

        # Select a random block of the time series.
        tmn, tmx = ds.time.min(), ds.time.max() - dt
        if tmn >= tmx:
            continue

        t0 = tmn + (tmx - tmn) * np.random.rand()
        m = (ds.time > t0) * (ds.time < t0 + dt)
        data.append((ds.time[m], ds.flux[m] - 1, ds.ferr[m]))

    # Initialize the estimate object (hack for multiprocessing).
    estimate = _stellar_estimate(data)

    # Send off a grid of optimizations.
    grid = [(1e-5, p) for p in np.linspace(1, 15, 16)]
    pool = Pool()
    results = sorted(pool.map(estimate, grid), key=lambda r: r.fun)

    # Update the hyper-parameters of the datasets.
    pars = results[0].x
    for ds in datasets:
        ds.hyperpars = pars

    return pars, data


class _stellar_estimate(object):

    def __init__(self, datasets):
        self.datasets = datasets

    def __call__(self, p0):
        return op.minimize(self.loss, p0, jac=True, method="L-BFGS-B",
                           bounds=[(0, None), (0, None)])

    def loss(self, p):
        ll, g = zip(*[_george.gradlnlikelihood(d[0], d[1], d[2], p[0], p[1])
                    for d in self.datasets])
        if np.all(np.isfinite(ll)):
            return -np.sum(ll), -sum(g)
        return np.inf, -sum(g)

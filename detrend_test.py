#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = []

import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as pl

import george

import bart
from bart.dataset import KeplerDataset


def robust_polyfit(x, y, yerr=None, order=3, Q=1.5):
    if yerr is None:
        yerr = np.ones_like(y)
    A = np.vstack([x ** i for i in range(order + 1)[::-1]]).T
    inds = np.ones_like(x, dtype=bool)
    for i in range(10):
        a = np.linalg.lstsq(A[inds, :] / yerr[inds][:, None],
                            y[inds] / yerr[inds])[0]
        p = np.poly1d(a)
        delta = (y - p(x)) ** 2 / yerr ** 2
        sigma = np.median(delta[inds])
        inds = delta < Q * sigma
    return p, inds


def robust_spline(x, y, yerr=None, Q=4):
    if yerr is None:
        yerr = np.ones_like(y)

    inds = np.argsort(x)
    x, y, yerr = x[inds], y[inds], yerr[inds]
    w = 1. / yerr / yerr
    t = np.arange(x[0], x[-1], 1)[1:]

    inds = np.ones_like(x, dtype=bool)
    for i in range(6):
        ti = (t > x[inds].min()) * (t < x[inds].max())
        p = LSQUnivariateSpline(x[inds], y[inds], t[ti], w=w[inds], k=3)
        delta = (y - p(x)) ** 2 / yerr ** 2
        sigma = np.median(delta[inds])
        inds = delta < Q * sigma

    return p, inds


def fit_gp(x, y, yerr=None):
    p = george.GaussianProcess([0.5, 1.])
    p.fit(x, y, yerr=yerr)
    return p


if __name__ == "__main__":
    api = bart.kepler.API()
    fns = api.data("10874614").fetch_all("kepler6data")

    for i, fn in enumerate(fns):
        pl.clf()
        ds = KeplerDataset(fn)

        x, y, yerr = ds.time, ds.flux, ds.ferr

        p, inds = robust_spline(x, y, yerr=yerr)
        gp = fit_gp(x, y, yerr=yerr)
        m, v = gp.predict(x, full_cov=False)
        print(m.shape, v.shape)

        pl.plot(x, y, "+k")
        pl.plot(x[inds], y[inds], ".k")
        pl.plot(x, p(x))
        pl.plot(x, m, "--r")
        pl.plot(x, m + np.sqrt(v), ":r")
        pl.plot(x, m - np.sqrt(v), ":r")
        pl.title(fn.replace("_", r"\_"))
        pl.savefig("detrend_test/{0:03d}.png".format(i))
        assert 0

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import os
import numpy as np
import matplotlib.pyplot as pl

import bart
from bart.dataset import KeplerDataset
from bart import kepler


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


if __name__ == "__main__":
    try:
        os.makedirs("detrend_test/results")
    except os.error:
        pass

    api = bart.kepler.API()
    fns = api.data("10593626").fetch_all("detrend_test/data")
    # fns = api.data("10874614").fetch_all("kepler6data")

    pl.figure(figsize=(8, 8))
    for i, fn in enumerate(fns):
        print(i)
        pl.clf()
        ds = KeplerDataset(fn, detrend=True)

        x, y, yerr = ds.time, ds.flux, ds.ferr
        p, t = kepler.spline_detrend(x, y, yerr=yerr, full_output=True, dt=2)

        pl.subplot(211)
        pl.plot(x, y, "+k")
        pl.plot(x, p(x), "r")
        pl.plot(t, p(t), ".r")
        pl.subplot(212)
        pl.plot(x, y - p(x), "+k")
        pl.title(fn.replace("_", r"\_"))
        pl.savefig("detrend_test/results/{0:03d}.png".format(i))
        pl.show()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import matplotlib.pyplot as pl

from bart import kepler
from bart.dataset import KeplerDataset


if __name__ == "__main__":
    api = kepler.API()

    koi = api.kois(kepid=6869184)[0]
    P = float(koi["Period"])
    t0 = float(koi["Time of Transit Epoch"]) % P

    fns = api.data("6869184").fetch_all("detrend_test/data")

    pl.figure(figsize=(8, 10))
    ax1 = pl.subplot(211)
    ax2 = pl.subplot(212)
    for fn in fns:
        ds = KeplerDataset(fn, detrend=True)
        ax2.plot(ds.time % P, ds.flux, ".k", alpha=0.3)
        ds = KeplerDataset(fn, detrend=False, kepler_detrend=True)
        ax1.plot(ds.time % P, ds.flux, ".k", alpha=0.3)

    ax1.set_ylim(0.99, 1.01)
    ax2.set_ylim(0.99, 1.01)

    # ax1.set_xlim(t0 - 1, t0 + 1)
    # ax2.set_xlim(t0 - 1, t0 + 1)

    pl.savefig("us_vs_them.png")

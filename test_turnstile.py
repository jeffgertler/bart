#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

from bart.injection import kepler_injection
from bart._turnstile import period_search
import numpy as np
import matplotlib.pyplot as pl

period, size = 278.1045694, 0.05
datasets, ps = kepler_injection(2301306, period, size, t0=20.0)

[pl.plot(d.time, d.flux, ".k") for d in datasets]
pl.savefig("data.png")

periods, epochs, depths, dvar = period_search(datasets,
                                              period - 0.01, period + 0.01, 5,
                                              100.0, 4.0)

mu = [np.mean(d) for d in depths]
print(mu)
pl.clf()
[pl.errorbar(p * np.ones_like(d), d, yerr=np.sqrt(e), fmt=".k")
 for p, d, e in zip(periods, depths, dvar)]
pl.plot(periods, mu, ".r")
pl.gca().axhline(size * size)
pl.gca().axvline(period)
pl.savefig("periods.png")

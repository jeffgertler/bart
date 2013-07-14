#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                "..", ".."))

import bart
from bart import _george
from bart.injection import kepler_injection
import numpy as np
import matplotlib.pyplot as pl

datasets, ps = kepler_injection(2301306, 400.0, 0.0)
pars, data = bart.utils.estimate_gp_hyperpars(datasets)
print(pars)

# Plot predictions.
nx = 5
ny = int(np.ceil(len(datasets) / nx))
print(nx, ny, len(datasets))
fig = pl.figure(figsize=(nx * 2, ny * 2))

for i, ds in enumerate(data):
    ax = fig.add_subplot(nx, ny, i + 1)
    t = np.linspace(ds[0].min(), ds[0].max(), 100)
    mu, cov = _george.predict(ds[0], ds[1], ds[2], pars[0], pars[1], t)
    model = np.random.multivariate_normal(mu, cov, size=50)
    ax.plot(ds[0], ds[1], ".k")
    ax.plot(t, model.T, "k", alpha=0.1)

fig.savefig("prediction.png")

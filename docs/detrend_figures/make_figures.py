#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib.pyplot as pl
from bart import KeplerDataset
from bart.kepler import spline_detrend


# Load the data.
fn = "kplr010874614-2009259160929_llc.fits"
ds = KeplerDataset(fn, detrend=False)
pdc_ds = KeplerDataset(fn, detrend=False, kepler_detrend=True)

# Do the de-trending.
p = spline_detrend(ds.time, ds.flux, yerr=ds.ferr)
factor = p(ds.time)

# Compute the axes limits.
xlim = (ds.time.min(), ds.time.max())

# Set up the figure.
fig = pl.figure(figsize=[8, 9])
ax = [fig.add_subplot(3, 1, i + 1) for i in range(3)]

# Plot the raw data.
ax[0].plot(ds.time, ds.flux, ".k", alpha=0.3)

# Plot the PDC data.
ax[1].plot(pdc_ds.time, pdc_ds.flux, ".k", alpha=0.3)

# Plot the de-trended data.
ax[2].plot(ds.time, ds.flux / factor, ".k", alpha=0.3)

# Set the limits and format the axes.
[a.set_xlim(xlim) for a in ax]
[a.set_xticklabels([]) for a in ax[:-1]]
ax[-1].set_xlabel("Time [KBJD]")

# Add annotations.
[a.annotate(s, [1, 1], xycoords="axes fraction", ha="right", va="top",
            xytext=[-5, -5], textcoords="offset points")
        for (a, s) in zip(ax, ["raw", "PDC corrected", "spline detrended"])]

fig.savefig("detrend.png")

ps = [spline_detrend(ds.time, ds.flux, yerr=ds.ferr, fill_times=False,
                     maxditer=1),
      spline_detrend(ds.time, ds.flux, yerr=ds.ferr, maxditer=1),
      spline_detrend(ds.time, ds.flux, yerr=ds.ferr)]

for i, p in enumerate(ps):
    # Plot the fitting figures.
    fig2 = pl.figure(figsize=[8, 6])
    ax2 = [fig2.add_subplot(2, 1, j + 1) for j in range(2)]

    # Do the most basic de-trending.
    factor2 = p(ds.time)

    # Plot the basic fit.
    t = np.linspace(xlim[0], xlim[1], 10 * len(ds.time))
    ax2[0].plot(ds.time, ds.flux, ".k", alpha=0.3)
    ylim = ax2[0].get_ylim()
    ax2[0].plot(t, p(t), "r", alpha=0.8)
    ax2[0].plot(p.get_knots(), p(p.get_knots()), ".r", alpha=1)
    ax2[0].set_ylim(ylim)

    # Plot the basic de-trend.
    ax2[1].plot(ds.time, ds.flux / factor2, ".k", alpha=0.3)

    # Format the axes.
    [a.set_xlim(xlim) for a in ax2]
    [a.set_xticklabels([]) for a in ax2[:-1]]
    ax2[-1].set_xlabel("Time [KBJD]")

    [a.annotate(s, [1, 1], xycoords="axes fraction", ha="right", va="top",
                xytext=[-5, -5], textcoords="offset points")
            for (a, s) in zip(ax2, ["raw",
                                    "spline detrending v{0}".format(i + 1)])]

    fig2.savefig("detrend_{0}.png".format(i + 2))

# Plot the zoomed figures.
fig3 = pl.figure(figsize=[8, 6])
ax3 = fig3.add_subplot(111)
chi = (ds.flux - ps[1](ds.time)) / ds.ferr
ax3.plot(ds.time, np.sqrt(4. / (4. + chi * chi)) * chi, ".k", alpha=0.3)
ax3.set_xlim(236, 255)
ax3.set_xlabel("Time [KBJD]")
fig3.savefig("detrend_5.png")

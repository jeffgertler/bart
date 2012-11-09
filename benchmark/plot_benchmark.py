#!/usr/bin/env python


import sys
import numpy as np
import matplotlib.pyplot as pl


data = np.array([line.split() for line in open(sys.argv[1])], dtype=float)
nb, nr = data[::10, 0], data[:10, 1]

# Time scalings.
fig = pl.figure(figsize=(6, 8))

ax = fig.add_axes([0.15, 0.6, 0.8, 0.35])

# ax.plot(np.log10(nb), np.log10(data[::10, 2]), ".k")
for i in range(len(nb)):
    ax.plot(np.log10(nr), np.log10(data[i:i + 10, 3]), "+k")

p = np.polyfit(np.log10(nb), np.log10(data[::10, 2]), 1)
ax.set_title(r"$\Delta_\mathrm{{rel}} \propto n_b^{{{0}}}$".format(p[0]))

ax.set_ylabel(r"$\log_{10} \, t \, [\mu \mathrm{s}]$")
ax.set_xlabel(r"$\log_{10} \, n_r$")

ax = fig.add_axes([0.15, 0.1, 0.8, 0.35])
ax.plot(np.log10(data[:, 1]), np.log10(data[:, 4]), ".k")

p = np.polyfit(np.log10(data[:, 1]), np.log10(data[:, 4]), 1)
ax.set_title(r"$\Delta_\mathrm{{rel}} \propto n_r^{{{0}}}$".format(p[0]))

ax.set_ylabel(r"$\log_{10} \, \Delta_\mathrm{rel}$")
ax.set_xlabel(r"$\log_{10} \, n_r$")

fig.savefig("scalings.pdf")

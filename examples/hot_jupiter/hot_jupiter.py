#!/usr/bin/env python

"""
A hot Jupiter dataset based _roughly_ on:
                        http://en.wikipedia.org/wiki/51_Pegasi_b

"""

import os
import sys

import scipy.optimize as op
from scipy.signal.spectral import lombscargle
import numpy as np
import matplotlib.pyplot as pl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
                                                os.path.abspath(__file__)))))
from bart import BART, fit_lightcurve


def box_model(p, T, t):
    D, L, P = p ** 2
    t0 = (t - P) % T
    return D * (t0 < L)


def fit_box(T, t, f, ferr):
    f -= np.mean(f)
    f /= np.min(f)
    p0 = [1.0, 0.01, np.pi]
    chi2 = lambda p: np.sum(((f - box_model(p, T, t)) / ferr) ** 2)
    result = op.minimize(chi2, p0)
    print result


if __name__ == "__main__":
    ps = BART(1.0, 100.0, 0.0, ldp=[500, 0.39, 0.1], ldptype="quad")
    ps.add_planet(5.0, 0.05, 0.013, 4.23, np.pi, 0.01)
    truth = ps.to_vector()

    N = 5000
    t = 1000. * np.random.rand(N)
    ferr = 5 * np.random.rand(N)
    f = ps.lightcurve(t) + ferr * np.random.randn(N)

    fit_box(ps.tp[0], t, f, ferr)
    # pl.errorbar(t % 4.23, f, yerr=ferr, fmt=".k")
    # pl.xlim(1.95, 2.3)
    # pl.savefig("data.png")

    # m = fit_lightcurve(t, f, ferr, T=4.0)
    # for i in range(len(truth)):
    #     pl.clf()
    #     pl.hist(m.flatchain[:, i], 100, color="k", histtype="step")
    #     pl.gca().axvline(truth[i])
    #     pl.title("Dimension {0:d}".format(i))
    #     pl.savefig("{0:02d}.png".format(i))

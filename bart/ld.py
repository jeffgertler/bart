#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["LimbDarkening", "QuadraticLimbDarkening"]

import numpy as np


class LimbDarkening(object):

    def __init__(self, bins, intensity):
        self.bins = np.array(bins)
        self.intensity = np.array(intensity)

    @property
    def norm(self):
        norm = np.pi * np.sum([intensity * (self.bins[i + 1] ** 2
                                            - self.bins[i] ** 2)
                            for i, intensity in enumerate(self.intensity[1:])])
        norm += np.pi * self.intensity[0] * self.bins[0] ** 2
        return norm

    def plot(self):
        x = [(0, self.bins[0])] + [(self.bins[i], self.bins[i + 1])
                                   for i in range(len(self.bins) - 1)]
        y = [(i, i) for i in self.intensity]
        return np.array(x), np.array(y) / self.norm


class QuadraticLimbDarkening(LimbDarkening):

    def __init__(self, gamma1, gamma2):
        self.gamma1, self.gamma2 = gamma1, gamma2

    @property
    def norm(self):
        return self.integrate(0, 1)

    def integrate(self, a, b):
        a2, b2 = a * a, b * b
        m1, m2 = self.gamma1, self.gamma2
        th = 0.5 * 3
        k1 = 0.5 * (b2 - a2) * (1 - m1 - 2 * m2)
        k2 = (m1 + 2 * m2) * ((1 - a2) ** th - (1 - b2) ** th) / 3
        k3 = 0.25 * m2 * (b2 * b2 - a2 * a2)
        return 2 * np.pi * (k1 + k2 + k3)

    def __call__(self, r):
        onemmu = 1 - np.sqrt(1 - r * r)
        return 1 - self.gamma1 * onemmu - self.gamma2 * onemmu * onemmu

    def histogram(self, bins):
        assert np.all(bins > 0) and np.all(bins <= 1)
        bins = np.append(0, bins)
        if bins[-1] < 1:
            bins = np.append(bins, 1)
        i = self.integrate(bins[:-1], bins[1:])
        i /= np.pi * (bins[1:] ** 2 - bins[:-1] ** 2)
        return LimbDarkening(bins[1:], i)


if __name__ == "__main__":
    # Check integral.
    ld = QuadraticLimbDarkening(0.25, 0.1)
    a, b = 0.0, 1.0
    analytic = ld.integrate(a, b)
    r = np.linspace(a, b, 5000000)
    i = ld(r)
    numerical = np.pi * np.sum(i[:-1] * (r[1:] ** 2 - r[:-1] ** 2))
    print(analytic, numerical)

    # Plot histograms.
    import matplotlib.pyplot as pl
    bins = np.linspace(0, 1, 20) ** 0.25
    ld2 = ld.histogram(bins[1:])
    x, y = ld2.plot()
    pl.plot(x.T, y.T, "k")
    pl.plot(r, i / ld.norm, "b")
    pl.savefig("ld-test.png")

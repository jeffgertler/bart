#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["LimbDarkening", "QuadraticLimbDarkening", "NonlinearLimbDarkening"]

import numpy as np


class LimbDarkening(object):

    def __init__(self, bins, intensity):
        self.bins = np.array(bins)
        self.intensity = np.array(intensity)

    def plot(self):
        x = [0, ]
        [(x.append(b), x.append(b)) for b in self.bins]
        y = []
        [(y.append(i), y.append(i)) for i in self.intensity]

        return x[:-1], y


class QuadraticLimbDarkening(LimbDarkening):

    def __init__(self, nbins, gamma1, gamma2):
        dr = 1.0 / nbins
        self.bins = np.arange(0, 1, dr) + dr
        self.gamma1, self.gamma2 = gamma1, gamma2

    def __getitem__(self, k):
        return getattr(self, k)

    def __setitem__(self, k, v):
        setattr(self, k, v)

    @property
    def intensity(self):
        onemmu = 1 - np.sqrt(1 - self.bins * self.bins)
        return 1 - self.gamma1 * onemmu - self.gamma2 * onemmu * onemmu


class NonlinearLimbDarkening(LimbDarkening):

    def __init__(self, nbins, coeffs):
        dr = 1.0 / nbins
        self.bins = np.arange(0, 1, dr) + dr
        self.coeffs = coeffs

    @property
    def intensity(self):
        mu = np.sqrt(1 - self.bins ** 2)
        c = self.coeffs
        return 1 - sum([c[i] * (1.0 - mu ** (0.5 * (i + 1)))
                                            for i in range(len(c))])

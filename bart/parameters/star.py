#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["LimbDarkeningParameters"]

import numpy as np
from .base import MultipleParameter, LogParameter
from bart.ldp import LimbDarkening


class LimbDarkeningParameters(MultipleParameter, LogParameter):

    def __init__(self, bins, fiducial, eta=0.1):
        self.bins = bins
        self.fiducial = fiducial
        self.N = len(bins)
        self.eta2 = eta * eta
        super(LimbDarkeningParameters, self).__init__(
                [r"$\log\,I_{{{0}}}$".format(i + 1) for i in range(self.N)])

    def __len__(self):
        return self.N

    def getter(self, star):
        intensity = star.ldp.intensity
        assert len(intensity) == self.N
        return np.array(intensity)

    def setter(self, star, vec):
        star.ldp = LimbDarkening(self.bins, vec)

    def lnprior(self, star):
        if np.any(star.ldp.intensity <= 0.0):
            return -np.inf
        lnp = np.sum((star.ldp.intensity - self.fiducial) ** 2) / self.eta2
        return -0.5 * lnp


class RelativeLimbDarkeningParameters(MultipleParameter):

    def __init__(self, bins):
        self.N = len(bins) - 1
        self.bins = bins
        super(RelativeLimbDarkeningParameters, self).__init__(
                [r"$\log\,\Delta I_{{{0}}}$".format(i + 1)
                 for i in range(self.N)])

    def __len__(self):
        return self.N

    def getter(self, star):
        intensity = star.ldp.intensity
        assert len(intensity) == self.N + 1
        return np.array([intensity[i] - intensity[i + 1]
                        for i in range(self.N)])

    def setter(self, star, vec):
        ldp = np.empty(self.N + 1)
        ldp[0] = 1
        for i, v in enumerate(vec):
            ldp[i + 1] = ldp[i] - v
        star.ldp = LimbDarkening(self.bins, ldp)

    def lnprior(self, star):
        d = star.ldp.intensity[1:] - star.ldp.intensity[:-1]
        if np.any(star.ldp.intensity <= 0.0) or np.any(d > 0):
            return -np.inf
        return 0.0

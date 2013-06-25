#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Prior", "UniformPrior", "NormalPrior"]

import numpy as np


class Prior(object):

    def __call__(self, value):
        return 0.0


class UniformPrior(Prior):

    def __init__(self, mn, mx):
        self.mn = mn
        self.mx = mx

    def __call__(self, value):
        if not self.mn < value < self.mx:
            return -np.inf
        return 0.0


class NormalPrior(Prior):

    def __init__(self, mu, sig2):
        self.mu = mu
        self._sig2 = None
        self._norm = None
        self.sig2 = sig2

    @property
    def sig2(self):
        return self._sig2

    @sig2.setter
    def sig2(self, v):
        self._sig2 = v
        self._norm = np.log(self.sig2)

    def __call__(self, value):
        d = self.mu - value
        return -0.5 * (d * d / self.sig2 + self._norm)

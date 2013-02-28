#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Prior", "UniformPrior", "GaussianPrior"]

import numpy as np


class Prior(object):

    def __call__(self, v):
        return 0.0

    def __repr__(self):
        return "Prior()"


class UniformPrior(Prior):

    def __init__(self, mn, mx):
        self.mn = mn
        self.mx = mx

    def __call__(self, v):
        if self.mn <= v <= self.mx:
            return 0.0
        return -np.inf

    def __repr__(self):
        return "UniformPrior({0.mn}, {0.mx})".format(self)


class GaussianPrior(Prior):

    def __init__(self, mu, std):
        self.mu = mu
        self.ivar = 1.0 / std / std

    def __call__(self, v):
        return -0.5 * np.sum(self.ivar * (v - self.mu) ** 2)

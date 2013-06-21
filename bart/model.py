#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "Model"]

import numpy as np


class Model(object):
    """
    A likelihood wrapper that combines a generative model and datasets to
    ...

    """

    def __init__(self, planetary_system, datasets=[]):
        self.planetary_system = planetary_system
        self.datasets = datasets
        self.parameters = []

    def add_parameter(self, parameter):
        self.parameters.append(parameter)

    @property
    def vector(self):
        return np.array([p.get(self) for p in self.parameters], dtype=float)

    @vector.setter
    def set_vector(self, values):
        [p.set(self, v) for p, v in zip(self.parameters, values)]

    def __call__(self, p):
        self.vector = p
        return self.lnprob()

    def lnprob(self):
        lp = self.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        ll = self.lnlike()
        if not np.isfinite(ll):
            return -np.inf
        return lp + ll

    def lnprior(self):
        return 0.0

    def lnlike(self):
        return np.sum([d.lnlike(self.planetary_system) for d in self.datasets])

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Model"]

import numpy as np


class Model(object):
    """
    A likelihood wrapper that combines a generative model and datasets to
    ...

    """

    def __init__(self, planetary_system, datasets=[], parameters=[],
                 priors=[]):
        self.planetary_system = planetary_system
        self.datasets = datasets
        self.parameters = parameters
        self.lnpriors = priors

    @property
    def vector(self):
        return np.array([p.get(self) for p in self.parameters], dtype=float)

    @vector.setter
    def vector(self, values):
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
        lp = self.planetary_system.lnprior()
        if not np.isfinite(lp):
            return -np.inf
        pp = [l(self) for l in self.lnpriors]
        if not np.all(np.isfinite(pp)):
            return -np.inf
        ppar = [p.lnprior(self) for p in self.parameters]
        if not np.all(np.isfinite(ppar)):
            return -np.inf
        return lp + np.sum(pp) + np.sum(ppar)

    def lnlike(self):
        return np.sum([d.lnlike(self) for d in self.datasets])

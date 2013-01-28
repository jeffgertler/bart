#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["EccentricityParameter"]

import numpy as np
from .base import MultipleParameter
from .priors import UniformPrior


class EccentricityParameter(MultipleParameter):

    def __init__(self):
        super(EccentricityParameter, self).__init__([r"$e\,\sin \varpi$",
                                                      r"$e\,\cos \varpi$"],
                                              priors=[UniformPrior(-1, 1),
                                                      UniformPrior(-1, 1)])

    def __repr__(self):
        return "EccentricityParameter()"

    def getter(self, obj):
        return np.array([obj.e * np.sin(obj.pomega),
                         obj.e * np.cos(obj.pomega)])

    def setter(self, obj, val):
        obj.e = np.sqrt(np.sum(val ** 2))
        obj.pomega = np.arctan2(val[0], val[1])

    def sample(self, obj, std=1e-5, size=1):
        e = np.abs(obj.e + std * np.random.randn(size))
        pomega = obj.pomega + std * np.random.randn(size)
        result = np.empty([2, size])
        result[0, :] = e * np.sin(pomega)
        result[1, :] = e * np.cos(pomega)
        return result

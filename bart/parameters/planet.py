#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["EccentricityParameter", "CosParameter"]

import numpy as np
from .base import MultipleParameter, Parameter


class EccentricityParameter(MultipleParameter):

    def __init__(self):
        super(EccentricityParameter, self).__init__([r"$e\,\sin\varpi$",
                                                     r"$e\,\cos\varpi$"])

    def __len__(self):
        return 2

    def getter(self, obj):
        return np.array([obj.e * np.sin(obj.pomega),
                         obj.e * np.cos(obj.pomega)])

    def setter(self, obj, val):
        obj.e = np.sqrt(np.sum(val ** 2))
        obj.pomega = np.arctan2(val[0], val[1])

    def sample(self, obj, std=1e-5, size=1):
        e = np.abs(obj.e + std * np.random.randn(size))
        pomega = 2 * np.pi * np.random.rand(size) - np.pi
        result = np.empty([2, size])
        result[0, :] = e * np.sin(pomega)
        result[1, :] = e * np.cos(pomega)
        return result

    def lnprior(self, obj):
        if 0 <= obj.e < 1 and -np.pi <= obj.pomega <= np.pi:
            return 0.0
        return -np.inf


class CosParameter(Parameter):

    def getter(self, obj):
        return np.cos(np.radians(obj.iobs))

    def setter(self, obj, val):
        obj.iobs = np.degrees(np.arccos(val))

    def sample(self, obj, std=1e-5, size=1):
        return np.cos(np.radians(obj.iobs * (1 + std * np.random.randn(size))))

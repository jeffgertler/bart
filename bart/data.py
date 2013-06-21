#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "Model"]

import numpy as np


class Parameter(object):
    """
    An abstract helper class that gets and sets model parameters for
    optimization or sampling.

    :param getter:
        A callable that takes (as input) a :class:`Model` object and returns
        the ``float`` value of the parameter.

    :param setter:
        A callable that takes a :class:`Model` and a ``float`` and sets the
        value of the parameter.

    :param conv: (optional)
        A conversion from the sampling coordinates and the physical
        coordinates.

    :param invconv: (optional)
        The inverse of :func:`conv`.

    """

    def __init__(self, getter, setter, conv=None, invconv=None):
        self.getter = getter
        self.setter = setter

        if conv is None:
            conv = lambda v: v
        self.conv = conv

        if invconv is None:
            invconv = lambda v: v
        self.invconv = invconv

    def get(self, model):
        return self.conv(self.getter(model))

    def set(self, model, value):
        return self.invconv(self.setter(model, value))


class Model(object):
    """
    A likelihood wrapper that combines a generative model and datasets to

    a change.

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

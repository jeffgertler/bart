#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "LogParameter", "MultipleParameter", "CosParameter"]

import numpy as np
from .priors import Prior


class Parameter(object):
    """
    A :class:`Parameter` is an object that gets and sets parameters in a
    model given a scalar value.

    :param name:
        The human-readable description of the parameter.

    :param attr:
        The name of the attribute to get and set.

    :param lnprior: (optional)
        A callable that computes the log-prior for the given parameter.

    """

    def __init__(self, name, attr, prior=None, plot_results=True):
        self.name = name
        self.attr = attr
        if prior is None:
            self.prior = Prior()
        else:
            self.prior = prior
        self.plot_results = plot_results

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{1}('{0.name}', attr='{0.attr}', prior={0.prior})".format(
            self, self.__class__.__name__)

    def __len__(self):
        return 1

    @property
    def names(self):
        return [self.name]

    def lnprior(self, obj):
        return self.prior(self.getter(obj))

    def sample(self, obj, std=1e-5, size=1):
        v0 = self.conv(self.getter(obj))
        if np.abs(v0) < std:
            return v0 + std * np.random.randn(size)
        return v0 * (1 + std * np.random.randn(size))

    def conv(self, val):
        """
        Convert from physical coordinates to fit coordinates.

        :param val:
            The physical parameter.

        """
        return val

    def iconv(self, val):
        """
        Convert from fit coordinates to physical coordinates.

        :param val:
            The fit parameter.

        """
        return val

    def getter(self, obj):
        """
        Get the fit value of this parameter from a given object.

        :param obj:
            The object that contains the parameter.

        """
        return getattr(obj, self.attr)

    def setter(self, obj, val):
        """
        Set the physical parameter(s) of the object ``obj`` given a particular
        fit parameter.

        :param obj:
            The object that contains the parameter.

        :param val:
            The value of the fit parameter to use.

        """
        setattr(obj, self.attr, val)


class LogParameter(Parameter):
    """
    A parameter that will be fit in log space.

    """

    def conv(self, val):
        return np.log(val)

    def iconv(self, val):
        return np.exp(val)


class MultipleParameter(Parameter):
    """
    A set of parameters that all rely on each other.

    :param names:
        List of names for each parameter.

    :param length:
        The number of fit parameters in the set.

    :param priors

    """

    def __init__(self, names, priors=None, plot_results=True):
        self._names = names
        if priors is None:
            self.priors = [Prior() for i in range(len(self.names))]
        else:
            self.priors = priors
        self.plot_results = plot_results

    def __str__(self):
        return "[" + ", ".join(self.names) + "]"

    def __repr__(self):
        return "{1}({0.names}, priors={0.priors})".format(
            self, self.__class__.__name__)

    def __len__(self):
        return len(self.names)

    def sample(self, obj, std=1e-5, size=1):
        v0 = self.conv(self.getter(obj))
        dim = len(v0)
        return v0[:, None] * (1 + std * np.random.randn(size * dim)
                                          .reshape(dim, size))

    def lnprior(self, obj):
        return np.sum([p(v) for p, v in zip(self.priors, self.getter(obj))])

    @property
    def names(self):
        return self._names

    def getter(self, obj):
        """
        Subclasses should implement this method.

        """
        raise NotImplementedError()

    def setter(self, obj, val):
        """
        Subclasses should implement this method.

        """
        raise NotImplementedError()


class CosParameter(Parameter):

    def getter(self, obj):
        return np.cos(np.radians(getattr(obj, self.attr)))

    def setter(self, obj, val):
        setattr(obj, self.attr, float(np.degrees(np.arccos(val))))

    def lnprior(self, obj):
        if 0 <= getattr(obj, self.attr) < 90.0:
            return 0.0
        return -np.inf

    def sample(self, obj, std=1e-5, size=1):
        i = getattr(obj, self.attr) * (1 + std * np.random.randn(size))
        while np.any(i > 90):
            i[i > 90] = 180 - i[i > 90]
        return np.cos(np.radians(i))

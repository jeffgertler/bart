#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "LogParameter", "MultipleParameter",
           "EccentricityParameters"]

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
        if self.mn < v < self.mx:
            return 0.0
        return -np.inf

    def __repr__(self):
        return "UniformPrior({0.mn}, {0.mx})".format(self)


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

    def __init__(self, name, attr, prior=None):
        self.name = name
        self.attr = attr
        if prior is None:
            self.prior = Prior()
        else:
            self.prior = prior

    def __str__(self):
        return self.name

    def __repr__(self):
        return "{1}('{0.name}', attr='{0.attr}', prior={0.prior})" \
                                    .format(self, self.__class__.__name__)

    def __len__(self):
        return 1

    @property
    def names(self):
        return [self.name]

    def lnprior(self, obj):
        return self.prior(self.iconv(self.getter(obj)))

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
        return self.conv(getattr(obj, self.attr))

    def setter(self, obj, val):
        """
        Set the physical parameter(s) of the object ``obj`` given a particular
        fit parameter.

        :param obj:
            The object that contains the parameter.

        :param val:
            The value of the fit parameter to use.

        """
        setattr(obj, self.attr, self.iconv(val))


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

    def __init__(self, names, priors=None):
        self._names = names
        if priors is None:
            self.priors = [Prior() for i in range(len(self.names))]
        else:
            self.priors = priors

    def __str__(self):
        return "[" + ", ".join(self.names) + "]"

    def __repr__(self):
        return "MultipleParameter({0.names}, {0.length}, prior={0.priors})" \
                                                                .format(self)

    def __len__(self):
        return len(self.names)

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


class EccentricityParameters(MultipleParameter):

    def __init__(self):
        super(EccentricityParameters, self).__init__([r"$e\,\sin \varpi$",
                                                      r"$e\,\cos \varpi$"],
                                              priors=[UniformPrior(-1, 1),
                                                      UniformPrior(-1, 1)])

    def __repr__(self):
        return "EccentricityParameters()"

    def lnprior(self, obj):
        return np.sum([p(v) for p, v in zip(self.priors, self.getter(obj))])

    def getter(self, obj):
        return np.array([obj.e * np.sin(obj.pomega),
                         obj.e * np.cos(obj.pomega)])

    def setter(self, obj, val):
        obj.e = np.sqrt(np.sum(val ** 2))
        obj.pomega = np.arctan2(val[0], val[1])

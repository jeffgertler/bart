#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "LogParameter", "ImpactParameter", "PeriodParameter",
           "LogPeriodParameter"]

import collections
import numpy as np

from .priors import Prior, UniformPrior


class Parameter(object):
    """
    Specification for a model parameter. This object specifies getter and
    setter methods that will be applied to a specific target object (or list
    of targets). The most basic usage would look something like:

    .. code-block:: python

        import bart
        from bart.parameters import Parameter
        from bart.priors import UniformPrior

        planet = bart.Planet(0.01, 100)
        parameter = Parameter(planet, "a", lnprior=UniformPrior(0, 200))

    if you wanted to sample the semi-major axis of the planet's orbit.
    Subclasses can implement more sophisticated transformations (see
    :class:`LogParameter` and :class:`ImpactParameter` for examples).

    :param targets:
        The target objects to which the parameter is related.

    :param attr: (optional)
        For a simple parameter, this is the attribute name of the parameter
        in ``targets``.

    :param lnprior: (optional)
        A callable log-prior function that should take in a single scalar (the
        result of :func:`Parameter.get`) and return the natural logarithm of
        the prior probability function.

    :param context: (optional)
        A dictionary of other information that the parameter might need to
        know about.

    """

    def __init__(self, targets, attr=None, lnprior=None, context={}):
        if isinstance(targets, collections.Iterable):
            self.targets = targets
        else:
            self.targets = [targets]
        self.attr = attr
        if lnprior is None:
            lnprior = Prior()
        self._lnprior = lnprior
        self.context = {}

    def __str__(self):
        return "{0}".format(self.attr)

    def __unicode__(self):
        return self.__str__()

    def __repr__(self):
        return "<{0}({1})>".format(self.__class__.__name__,
                                   self.__str__())

    def get(self):
        return self.conv(self.getter())

    def set(self, value):
        self.setter(self.invconv(value))

    def getter(self):
        vals = [getattr(t, self.attr) for t in self.targets]
        assert all([v == vals[0] for v in vals])
        return vals[0]

    def setter(self, value):
        [setattr(t, self.attr, value) for t in self.targets]

    def conv(self, value):
        return value

    def invconv(self, value):
        return value

    def lnprior(self):
        return self._lnprior(self.get())


class LogParameter(Parameter):
    """
    Similar to :class:`Parameter` but the sampling will be performed in the
    logarithm of the physical parameter.

    """

    def __str__(self):
        nm = super(LogParameter, self).__str__()
        return r"\ln\,{0}".format(nm)

    def conv(self, value):
        return np.log(value)

    def invconv(self, value):
        return np.exp(value)


class ImpactParameter(Parameter):
    """
    A :class:`Parameter` subclass that makes it easy to sample the impact
    parameter of a planet.

    :param planet:
        The :class:`Planet` itself.

    """

    def __str__(self):
        return "b"

    def __init__(self, planet, **kwargs):
        lnprior = kwargs.pop("lnprior", None)
        if lnprior is None:
            lnprior = UniformPrior(0.0, 1.0)
        kwargs["lnprior"] = lnprior
        super(ImpactParameter, self).__init__(planet, **kwargs)

    def getter(self):
        iobs = self.targets[0].planetary_system.iobs
        return self.targets[0].a / np.tan(np.radians(iobs
                                                     - self.targets[0].ix))

    def setter(self, b):
        iobs = self.targets[0].planetary_system.iobs
        a = self.targets[0].a
        for t in self.targets:
            t.ix = iobs - np.degrees(np.arctan2(a, b))


class PeriodParameter(Parameter):
    """
    A :class:`Parameter` subclass that samples the period of a :class:`Planet`
    orbit instead of the impact parameter.

    :param planet:
        The :class:`Planet` itself.

    :param lnprior: (optional)
        A callable that returns the natural logarithm of the prior probability
        function given a period. This function should return ``-numpy.inf``
        for disallowed periods.

    """

    def __str__(self):
        return "P"

    def getter(self):
        smass = self.targets[0].planetary_system.star.mass
        return self.targets[0].get_period(smass)

    def setter(self, period):
        s = self.targets[0].planetary_system.star
        m = self.targets[0].mass
        for t in self.targets:
            t.a = s.get_semimajor(period, planet_mass=m)


class LogPeriodParameter(PeriodParameter, LogParameter):
    """
    The same as :class:`PeriodParameter` except the sampling is performed in
    natural logarithm of the period.

    """


# class MultiParameter(Parameter):

#     def getter(self):
#         return getattr(self.target[0], self.attr)

#     def setter(self, value):
#         [setattr(t, self.attr, value) for t in self.target]


# class LogMultiParameter(MultiParameter, LogParameter):

#     pass

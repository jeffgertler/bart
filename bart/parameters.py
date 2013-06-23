#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "LogParameter", "ImpactParameter", "PeriodParameter",
           "LogPeriodParameter"]


import numpy as np

from .priors import Prior, UniformPrior


class Parameter(object):
    """
    Specification for a model parameter. This object specifies getter and
    setter methods that will be applied to a specific target. The most basic
    usage would look something like:

    .. code-block:: python

        import bart
        from bart.parameters import Parameter
        from bart.priors import UniformPrior

        planet = bart.Planet(0.01, 100)
        parameter = Parameter(planet, "a", lnprior=UniformPrior(0, 200))

    if you wanted to sample the semi-major axis of the planet's orbit.
    Subclasses can implement more sophisticated transformations (see
    :class:`LogParameter` and :class:`ImpactParameter` for examples).

    :param target:
        The target object to which the parameter is related.

    :param attr: (optional)
        For a simple parameter, this is the attribute name of the parameter
        in ``target``.

    :param lnprior: (optional)
        A callable log-prior function that should take in a single scalar (the
        result of :func:`Parameter.get`) and return the natural logarithm of
        the prior probability function.

    :param context: (optional)
        A dictionary of other information that the parameter might need to
        know about.

    """

    def __init__(self, target, attr=None, lnprior=None, context={}):
        self.target = target
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
        return getattr(self.target, self.attr)

    def setter(self, value):
        return setattr(self.target, self.attr, value)

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
        return r"\ln\,{0}".format(self.attr)

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

    def __init__(self, planet, **kwargs):
        lnprior = kwargs.pop("lnprior", None)
        if lnprior is None:
            lnprior = UniformPrior(0.0, 1.0)
        super(ImpactParameter, self).__init__(planet, **kwargs)

    def getter(self):
        iobs = self.target.planetary_system.iobs
        return self.target.a / np.tan(np.radians(iobs - self.target.ix))

    def setter(self, b):
        iobs = self.target.planetary_system.iobs
        self.target.ix = iobs - np.degrees(np.arctan2(self.target.a, b))


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

    def getter(self):
        smass = self.target.planetary_system.star.mass
        return self.target.get_period(smass)

    def setter(self, period):
        s = self.target.planetary_system.star
        self.target.a = s.get_semimajor(period, planet_mass=self.target.mass)


class LogPeriodParameter(PeriodParameter, LogParameter):
    """
    The same as :class:`PeriodParameter` except the sampling is performed in
    natural logarithm of the period.

    """

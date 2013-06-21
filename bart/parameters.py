#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "PlanetParameter", "ImpactParameter",
           "PeriodParameter"]


import numpy as np

from .priors import Prior


class Parameter(object):

    def __init__(self, lnprior=None):
        if lnprior is None:
            lnprior = Prior()
        self._lnprior = lnprior

    def get(self, model):
        return self.conv(self.getter(model))

    def set(self, model, value):
        self.setter(model, self.invconv(value))

    def getter(self, model):
        raise NotImplementedError()

    def setter(self, model, value):
        raise NotImplementedError()

    def conv(self, value):
        return value

    def invconv(self, value):
        return value

    def lnprior(self, model):
        return self._lnprior(self.get(model))


class LogParameter(Parameter):

    def conv(self, value):
        return np.log(value)

    def invconv(self, value):
        return np.exp(value)


class PlanetParameter(Parameter):

    def __init__(self, attr, index=0, **kwargs):
        super(PlanetParameter, self).__init__(**kwargs)
        self.attr = attr
        self.index = index

    def getter(self, model):
        return getattr(model.planetary_system.planets[self.index], self.attr)

    def setter(self, model, val):
        setattr(model.planetary_system.planets[self.index], self.attr, val)


class ImpactParameter(Parameter):

    def __init__(self, index=0, **kwargs):
        super(ImpactParameter, self).__init__(**kwargs)
        self.index = index

    def getter(self, model):
        iobs = model.planetary_system.iobs
        p = model.planetary_system.planets[self.index]
        return p.a / np.tan(np.radians(iobs - p.ix))

    def setter(self, model, b):
        iobs = model.planetary_system.iobs
        p = model.planetary_system.planets[self.index]
        p.ix = iobs - np.degrees(np.arctan2(p.a, b))


class PeriodParameter(Parameter):

    def __init__(self, index=0, **kwargs):
        super(PeriodParameter, self).__init__(**kwargs)
        self.index = index

    def getter(self, model):
        smass = model.planetary_system.star.mass
        return model.planetary_system.planets[self.index].get_period(smass)

    def setter(self, model, period):
        p = model.planetary_system.planets[self.index]
        p.a = model.planetary_system.star.get_semimajor(period,
                                                        planet_mass=p.mass)

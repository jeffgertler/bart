#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter", "PlanetParameter", "ImpactParameter",
           "PeriodParameter"]


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

    def __init__(self, spec=None, getter=None, setter=None,
                 conv=None, invconv=None):
        if spec is not None:
            self.getter = lambda m: eval("model.{0}".format(spec),
                                         {"model": m})
            self.setter = lambda m, v: eval("model.{0} = value".format(spec),
                                            {"model": m, "value": v})

        else:
            assert setter is not None and getter is not None
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
        self.setter(model, self.invconv(value))


class PlanetParameter(Parameter):

    def __init__(self, attr, index=0, **kwargs):
        self.attr = attr
        self.index = index

        def getter(model):
            return getattr(model.planetary_system.planets[index], attr)

        def setter(model, val):
            setattr(model.planetary_system.planets[index], attr, val)

        kwargs["getter"] = getter
        kwargs["setter"] = setter

        super(PlanetParameter, self).__init__(**kwargs)


class ImpactParameter(Parameter):

    def __init__(self, index=0, **kwargs):
        self.index = index

        def getter(model):
            iobs = model.planetary_system.iobs
            p = model.planetary_system.planets[index]
            return p.a / np.tan(np.radians(iobs - p.ix))

        def setter(model, b):
            iobs = model.planetary_system.iobs
            p = model.planetary_system.planets[index]
            p.ix = iobs - np.degrees(np.arctan2(p.a, b))

        kwargs["getter"] = getter
        kwargs["setter"] = setter

        super(ImpactParameter, self).__init__(**kwargs)


class PeriodParameter(Parameter):

    def __init__(self, index=0, **kwargs):
        self.index = index

        def getter(model):
            smass = model.planetary_system.star.mass
            return model.planetary_system.planets[index].get_period(smass)

        def setter(model, period):
            p = model.planetary_system.planets[index]
            p.a = model.planetary_system.star.get_semimajor(period,
                                                            planet_mass=p.mass)

        kwargs["getter"] = getter
        kwargs["setter"] = setter

        super(PeriodParameter, self).__init__(**kwargs)

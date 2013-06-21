#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Parameter"]


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
            self.getter = lambda m: eval(spec, {"model": m})
            self.setter = lambda m, v: eval("{0} = value".format(spec),
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
        return self.invconv(self.setter(model, value))

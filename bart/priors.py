#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Prior", "UniformPrior"]

import numpy as np


class Prior(object):

    def __call__(self, value):
        return 0.0


class UniformPrior(Prior):

    def __init__(self, mn, mx):
        self.mn = mn
        self.mx = mx

    def __call__(self, value):
        if not self.mn < value < self.mx:
            return -np.inf
        return 0.0

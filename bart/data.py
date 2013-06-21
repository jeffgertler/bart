#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Dataset", "LightCurve"]


import numpy as np
from . import _george


class Dataset(object):

    pass


class LightCurve(Dataset):
    """
    Wrapper around a light curve dataset. This does various nice things like
    masking NaNs and Infs and normalizing the fluxes by the median.

    :param time:
        The time series in days.

    :param flux:
        The flux measurements in arbitrary units.

    :param ferr:
        The error bars on ``flux``.

    :param texp: (optional)
        The integration time (in seconds). (default: 1626.0… Kepler
        long-cadence)

    :param K: (optional)
        The number of bins to use in the approximate exposure time integral.
        (default: 3)

    :param alpha: (optional)
        The amplitude of the GP kernel. (default: 1.0)

    :param l2: (optional)
        The variance scale of the GP. (default: 3.0)

    """

    def __init__(self, time, flux, ferr, texp=1626.0, K=3, alpha=1.0, l2=3.0):
        m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(ferr)
        self.time = time[m]
        self.flux = flux[m]
        self.ferr = ferr[m]

        # Normalize by the median.
        mu = np.median(self.flux)
        self.flux /= mu
        self.ferr /= mu

        # Light curve parameters.
        self.texp = texp
        self.K = K

        # Gaussian process parameters.
        self.alpha = alpha
        self.l2 = l2

    def lnlike(self, model):
        lc = model.planetary_system.lightcurve(self.time, texp=self.texp,
                                               K=self.K)
        return _george.lnlikelihood(self.time, self.flux / lc - 1, self.ferr,
                                    self.alpha, self.l2)

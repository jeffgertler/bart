#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["Dataset", "LightCurve", "GPLightCurve", "PhotonStream"]


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

    """

    def __init__(self, time, flux, ferr, texp=1626.0, K=3):
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

    def lnlike(self, model):
        """
        Get the likelihood of this dataset given a particular :class:`Model`.

        :param model:
            The :class:`Model` specifying the model to compare the data to.

        """
        lc = model.planetary_system.lightcurve(self.time, texp=self.texp,
                                               K=self.K)
        return np.sum(-0.5 * (lc - self.flux) ** 2)


class GPLightCurve(LightCurve):
    """
    An extension to :class:`LightCurve` with a Gaussian Process likelihood
    function. This does various nice things like masking NaNs and Infs and
    normalizing the fluxes by the median.

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

    def __init__(self, time, flux, ferr, alpha=1.0, l2=3.0, **kwargs):
        super(GPLightCurve, self).__init__(time, flux, ferr, **kwargs)
        self.alpha = alpha
        self.l2 = l2

    def lnlike(self, model):
        """
        Get the likelihood of this dataset given a particular :class:`Model`.

        :param model:
            The :class:`Model` specifying the model to compare the data to.

        """
        lc = model.planetary_system.lightcurve(self.time, texp=self.texp,
                                               K=self.K)
        return _george.lnlikelihood(self.time, self.flux / lc - 1, self.ferr,
                                    self.alpha, self.l2)


class PhotonStream(Dataset):
    """
    An extension to :class:`LightCurve` with a Poisson likelihood function.
    This class automatically masks all NaNs in the data stream.

    :param time:
        The times of the samples in days.

    :param dt: (optional)
        The bin size in days. (default: 0.1)

    """

    def __init__(self, time, dt=0.1, K=3):
        self.time = time[np.isfinite(time)]
        self.dt = dt

    def lnlike(self, model):
        """
        Get the likelihood of this dataset given a particular :class:`Model`.

        :param model:
            The :class:`Model` specifying the model to compare the data to.

        """
        photonrates = self.rate(model, self.time)
        bintimes = np.arange(self.time.min(), self.time.max(), self.dt)
        binrates = self.rate(model, bintimes)
        prob = np.sum(np.log(photonrates)) - self.dt * np.sum(binrates)
        return prob

    def rate(self, model, t):
        """
        Return the rate estimate for a set of time samples. This function
        combines the estimated light curve, the sensitivity function and the
        background rate.

        :param model:
            The :class:`Model` to compute the estimate for.

        :param t:
            The time points in days.

        """
        lc = model.planetary_system.lightcurve(t, texp=0, K=1)
        return lc * self.sensitivity(t) + self.background(t)

    def background(self, t):
        """
        The background function. The default implementation is a trivial zero
        level background. Subclasses should overload this when they have a
        better understanding of the instrument.

        :param t:
            The time points in days.

        """
        return np.zeros_like(t)

    def sensitivity(self, t):
        """
        The sensitivity function of the instrument. The default implementation
        simply returns an array of ones so subclasses should return something
        more sophisticated.

        :param t:
            The time points in days.

        """
        return np.ones_like(t)

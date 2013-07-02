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

    def __init__(self, time, flux, ferr, texp=1626.0, K=3, dtbin=None):
        m = np.isfinite(time) * np.isfinite(flux) * np.isfinite(ferr)

        if dtbin is not None:
            tmn, tmx = np.min(time), np.max(time)
            time, flux, ferr = time[m], flux[m], ferr[m]
            ivar = 1.0 / (ferr * ferr)

            self.time = np.arange(tmn, tmx + dtbin, dtbin)
            self.flux = np.zeros_like(self.time)
            self.ivar = np.zeros_like(self.time)
            bind = np.floor((time - tmn) / dtbin)
            for i in range(len(self.time)):
                m = bind == i
                if np.any(m):
                    self.ivar[i] = np.sum(ivar[m])
                    self.flux[i] = np.sum(flux[m] * ivar[m]) / self.ivar[i]

            m = self.ivar > 0
            self.time = self.time[m]
            self.flux = self.flux[m]
            self.ivar = self.ivar[m]
            self.ferr = 1.0 / np.sqrt(self.ivar)

        else:
            self.time = time[m]
            self.flux = flux[m]
            self.ferr = ferr[m]
            self.ivar = 1.0 / (self.ferr * self.ferr)

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
        lc = self.predict(model)
        return np.sum(-0.5 * (lc - self.flux) ** 2 * self.ivar)

    def predict(self, model, t=None):
        """
        Generate a model light curve for a particular :class:`Model`.

        :param model:
            The :class:`Model` specifying the model to generate from.

        :param t: (optional)
            The times where the model should be evaluated. By default, it'll
            return the light curve evaluated at the data points.

        """
        if t is None:
            t = self.time
        return model.planetary_system.lightcurve(t, texp=self.texp,
                                                 K=self.K)


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

    def predict(self, model, t=None):
        """
        Generate a sample from the light curve probability function for a
        particular :class:`Model`.

        :param model:
            The :class:`Model` specifying the model to generate from.

        :param t: (optional)
            The times where the model should be evaluated. By default, it'll
            return the light curve evaluated at the data points.

        """
        if t is None:
            t = self.time
            lc0 = model.planetary_system.lightcurve(t, texp=self.texp,
                                                    K=self.K)
            lc = lc0
        else:
            lc0 = model.planetary_system.lightcurve(self.time, texp=self.texp,
                                                    K=self.K)
            lc = model.planetary_system.lightcurve(t, texp=self.texp,
                                                   K=self.K)

        mu, cov = _george.predict(self.time, self.flux / lc0 - 1, self.ferr,
                                  self.alpha, self.l2, t)
        return (np.random.multivariate_normal(mu, cov) + 1) * lc


class PhotonStream(Dataset):
    """
    An extension to :class:`LightCurve` with a Poisson likelihood function.
    This class automatically masks all NaNs in the data stream.

    :param time:
        The times of the samples in days.

    :param dt: (optional)
        The bin size in days. (default: 0.1)

    """

    def __init__(self, time, dt=0.1, background=0.0):
        self.time = time[np.isfinite(time)]
        self.dt = dt
        self.bglevel = background

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
        return np.zeros_like(t) + self.bglevel

    def sensitivity(self, t):
        """
        The sensitivity function of the instrument. The default implementation
        simply returns an array of ones so subclasses should return something
        more sophisticated.

        :param t:
            The time points in days.

        """
        return np.ones_like(t)

    def predict(self, model, t=None):
        """
        Generate a model light curve for a particular :class:`Model`.

        :param model:
            The :class:`Model` specifying the model to generate from.

        :param t: (optional)
            The times where the model should be evaluated. By default, it'll
            return the light curve evaluated at the data points.

        """
        raise NotImplementedError()

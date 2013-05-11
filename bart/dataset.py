#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (division, print_function, absolute_import,
                        unicode_literals)

__all__ = ["LCDataset", "KeplerDataset", "RVDataset"]

import pyfits
import numpy as np
from .bart import Model
from . import kepler


class LCDataset(Model):
    """
    Wrapper around a light curve dataset.

    :param time:
        An array of timestamps. We don't make any assumptions about the units
        of these times but it's probably best to stick to KBJD.

    :param flux:
        The array of flux measurements corresponding to the time samples.
        Again, the units of this array are arbitrary.

    :param ferr:
        The array of observed error bars on the flux measurements in the same
        units as ``flux``.

    :param texp:
        The exposure time in the same units as ``time``.

    :param zp: (optional)
        The multiplicative "zero point" of the flux scale. (default: 1.0)

    :param jitter: (optional)
        A constant additive variance allowing ``ferr ** 2`` to include for
        underestimated error bars across the whole dataset. Should be in the
        same units as ``flux``. (default: 0.0)

    """

    __type__ = "lc"

    def __init__(self, time, flux, ferr, texp, zp=1.0, jitter=0.0):
        super(LCDataset, self).__init__()

        self.texp = texp
        self.jitter = jitter
        self.zp = zp

        # Sanitize the data.
        inds = ~np.isnan(time) * ~np.isnan(flux) * ~np.isnan(ferr)
        self.time, self.flux, self.ferr = time[inds], flux[inds], ferr[inds]
        self.ivar = 1.0 / self.ferr / self.ferr

    def __len__(self):
        return len(self.time)


class KeplerDataset(LCDataset):
    """
    Wrapper around a light curve dataset from Kepler.

    :param time:
        An array of timestamps in KBJD.

    :param flux:
        The array of flux measurements corresponding to the time samples.

    :param ferr:
        The array of observed error bars on the flux measurements in the same
        units as ``flux``.

    :param zp: (optional)
        The multiplicative "zero point" of the flux scale. (default: 1.0)

    :param jitter: (optional)
        A constant additive variance allowing ``ferr ** 2`` to include for
        underestimated error bars across the whole dataset. Should be in the
        same units as ``flux``. (default: 0.0)

    :param detrend: (optional)
        Should the dataset be automatically de-trended using the spline
        de-trending algorithm? (default: True)

    :param kepler_detrend: (optional)
        Should we start with the PDC calibrated flux measurements instead of
        the raw SAP fluxes? (default: False)

    """

    def __init__(self, fn, zp=1.0, jitter=0.0, detrend=True,
                 kepler_detrend=False):
        f = pyfits.open(fn)
        lc = np.array(f[1].data)
        self.cadence = 0 if f[0].header["OBSMODE"] == "short cadence" else 1
        f.close()

        # Get the exposure time.
        # http://archive.stsci.edu/mast_faq.php?mission=KEPLER#50
        texp = kepler.EXPOSURE_TIMES[self.cadence] / 86400.

        time = lc["TIME"]
        if kepler_detrend:
            flux, ferr = lc["PDCSAP_FLUX"], lc["PDCSAP_FLUX_ERR"]
        else:
            flux, ferr = lc["SAP_FLUX"], lc["SAP_FLUX_ERR"]

        super(KeplerDataset, self).__init__(time, flux, ferr, texp,
                                            jitter=jitter)

        # Remove the arbitrary median.
        self.median = np.median(self.flux)
        self.flux /= self.median
        self.ferr /= self.median
        self.ivar *= self.median * self.median

        if detrend:
            p = kepler.spline_detrend(self.time, self.flux, self.ferr)
            factor = p(self.time)
            self.flux /= factor
            self.ferr /= factor
            self.ivar *= factor * factor


class RVDataset(Model):
    """
    Wrapper around a radial velocity dataset.

    :param time:
        An array of timestamps. We don't make any assumptions about the units
        of these times but it should be in the same units as the rest of your
        datasets so it's probably best to stick to KBJD.

    :param rv:
        The array of radial velocity measurements in :math:`m\,s^{-1}`.

    :param rverr:
        The uncertainties on ``rv`` measured in :math:`m\,s^{-1}`.

    :param jitter: (optional)
        A constant additive variance allowing ``rverr ** 2`` to include for
        underestimated error bars across the whole dataset. Should be in
        :math:`m\,s^{-1}`. (default: 0.0)

    """

    __type__ = "rv"

    def __init__(self, time, rv, rverr, jitter=0.0):
        super(RVDataset, self).__init__()
        inds = ~np.isnan(time) * ~np.isnan(rv) * ~np.isnan(rverr)
        self.time = time[inds]
        self.rv = rv[inds]
        self.rverr = rverr[inds]
        self.ivar = 1.0 / self.rverr / self.rverr
        self.jitter = jitter
